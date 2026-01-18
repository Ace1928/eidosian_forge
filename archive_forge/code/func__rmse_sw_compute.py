from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.image.utils import _uniform_filter
from torchmetrics.utilities.checks import _check_same_shape
def _rmse_sw_compute(rmse_val_sum: Optional[Tensor], rmse_map: Tensor, total_images: Tensor) -> Tuple[Optional[Tensor], Tensor]:
    """Compute RMSE from the aggregated RMSE value. Optionally also computes the mean value for RMSE map.

    Args:
        rmse_val_sum: Sum of RMSE over all examples
        rmse_map: Sum of RMSE map values over all examples
        total_images: Total number of images

    Return:
        RMSE using sliding window
        (Optionally) RMSE map

    """
    rmse = rmse_val_sum / total_images if rmse_val_sum is not None else None
    if rmse_map is not None:
        rmse_map /= total_images
    return (rmse, rmse_map)