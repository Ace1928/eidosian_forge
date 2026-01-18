from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _get_normalized_sim_and_cs(preds: Tensor, target: Tensor, gaussian_kernel: bool=True, sigma: Union[float, Sequence[float]]=1.5, kernel_size: Union[int, Sequence[int]]=11, data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, normalize: Optional[Literal['relu', 'simple']]=None) -> Tuple[Tensor, Tensor]:
    sim, contrast_sensitivity = _ssim_update(preds, target, gaussian_kernel, sigma, kernel_size, data_range, k1, k2, return_contrast_sensitivity=True)
    if normalize == 'relu':
        sim = torch.relu(sim)
        contrast_sensitivity = torch.relu(contrast_sensitivity)
    return (sim, contrast_sensitivity)