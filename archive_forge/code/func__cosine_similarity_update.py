from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _cosine_similarity_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Cosine Similarity. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    if preds.ndim != 2:
        raise ValueError(f'Expected input to cosine similarity to be 2D tensors of shape `[N,D]` where `N` is the number of samples and `D` is the number of dimensions, but got tensor of shape {preds.shape}')
    preds = preds.float()
    target = target.float()
    return (preds, target)