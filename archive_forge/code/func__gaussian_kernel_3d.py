from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _gaussian_kernel_3d(channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Compute 3D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w, d)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    """
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    gaussian_kernel_z = _gaussian(kernel_size[2], sigma[2], dtype, device)
    kernel_xy = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)
    kernel = torch.mul(kernel_xy.unsqueeze(-1).repeat(1, 1, kernel_size[2]), gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]))
    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1], kernel_size[2])