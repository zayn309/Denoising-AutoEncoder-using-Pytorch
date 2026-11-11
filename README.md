# Denoising Autoencoder (PyTorch)

A convolutional **denoising autoencoder** that learns to remove noise from RGB images, trained with a combination of a pixel-level **Charbonnier loss** and a **VGG19 perceptual loss**.

All the work lives in [`denoising_autoencoder.ipynb`](denoising_autoencoder.ipynb), built to run on Kaggle with a GPU.

## How it works

- **Data:** clean images are loaded from a Kaggle image dataset, and noisy inputs are generated on the fly with **Albumentations** augmentations. Each training pair is *(noisy image → clean image)*.
- **Model:** a convolutional encoder compresses the image into a compact feature representation, and a decoder reconstructs the clean image from it.
- **Loss:**
  - **Charbonnier loss** (a smooth, robust L1) keeps the output close to the clean image pixel by pixel.
  - **VGG19 perceptual loss** compares deep features of the output and the clean image, which keeps textures and edges sharp instead of blurry.
  - `total = α · Charbonnier + β · VGG` with α = 1 and β = 0.05.
- **Training:** Adam (lr = 1e-3), batch size 32, 10 epochs, with periodic evaluation on a held-out split.

## Results

Validation loss dropped from **0.83** before training to about **0.30** within the first few epochs, and kept improving after that. Before/after examples are plotted at the end of the notebook.

## Getting started

```bash
pip install torch torchvision albumentations pillow tqdm numpy pandas
```

Open the notebook in Kaggle or Jupyter, set the image directory to your dataset, and run all cells.

## License

[MIT](LICENSE)
