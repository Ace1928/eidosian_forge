from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
def _spectral_distortion_index_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Spectral Distortion Index.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image

    """
    if preds.dtype != target.dtype:
        raise TypeError(f'Expected `ms` and `fused` to have the same data type. Got ms: {preds.dtype} and fused: {target.dtype}.')
    if len(preds.shape) != 4:
        raise ValueError(f'Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}.')
    if preds.shape[:2] != target.shape[:2]:
        raise ValueError(f'Expected `preds` and `target` to have same batch and channel sizes.Got preds: {preds.shape} and target: {target.shape}.')
    return (preds, target)