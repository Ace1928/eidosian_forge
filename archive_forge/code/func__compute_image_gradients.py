from typing import Tuple
import torch
from torch import Tensor
def _compute_image_gradients(img: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute image gradients (dy/dx) for a given image."""
    batch_size, channels, height, width = img.shape
    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]
    shapey = [batch_size, channels, 1, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
    dy = dy.view(img.shape)
    shapex = [batch_size, channels, height, 1]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
    dx = dx.view(img.shape)
    return (dy, dx)