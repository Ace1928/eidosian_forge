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
def _peak_signal_noise_ratio(preds: Tensor, target: Tensor, data_range: Optional[Union[float, Tuple[float, float]]]=None, base: float=10.0, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', dim: Optional[Union[int, Tuple[int, ...]]]=None) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> pred = tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> target = tensor([[3.0, 2.0], [1.0, 0.0]])
    >>> _peak_signal_noise_ratio(pred, target)
    tensor(2.5527)

    """
    _deprecated_root_import_func('peak_signal_noise_ratio', 'image')
    return peak_signal_noise_ratio(preds=preds, target=target, data_range=data_range, base=base, reduction=reduction, dim=dim)