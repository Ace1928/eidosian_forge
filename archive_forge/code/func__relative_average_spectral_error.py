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
def _relative_average_spectral_error(preds: Tensor, target: Tensor, window_size: int=8) -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> gen = torch.manual_seed(22)
    >>> preds = torch.rand(4, 3, 16, 16, generator=gen)
    >>> target = torch.rand(4, 3, 16, 16, generator=gen)
    >>> _relative_average_spectral_error(preds, target)
    tensor(5114.66...)

    """
    _deprecated_root_import_func('relative_average_spectral_error', 'image')
    return relative_average_spectral_error(preds=preds, target=target, window_size=window_size)