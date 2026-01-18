from typing import Optional, Sequence, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.rase import relative_average_spectral_error
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.functional.image.ssim import (
from torchmetrics.functional.image.tv import total_variation
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _image_gradients(img: Tensor) -> Tuple[Tensor, Tensor]:
    """Wrapper for deprecated import.

    >>> import torch
    >>> image = torch.arange(0, 1*1*5*5, dtype=torch.float32)
    >>> image = torch.reshape(image, (1, 1, 5, 5))
    >>> dy, dx = _image_gradients(image)
    >>> dy[0, 0, :, :]
    tensor([[5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [0., 0., 0., 0., 0.]])

    """
    _deprecated_root_import_func('image_gradients', 'image')
    return image_gradients(img=img)