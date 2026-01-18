from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
def _cosine_similarity_compute(preds: Tensor, target: Tensor, reduction: Optional[str]='sum') -> Tensor:
    """Compute Cosine Similarity.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        >>> preds = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]])
        >>> preds, target = _cosine_similarity_update(preds, target)
        >>> _cosine_similarity_compute(preds, target, 'none')
        tensor([ 1.0000, -1.0000])

    """
    dot_product = (preds * target).sum(dim=-1)
    preds_norm = preds.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    similarity = dot_product / (preds_norm * target_norm)
    reduction_mapping = {'sum': torch.sum, 'mean': torch.mean, 'none': lambda x: x, None: lambda x: x}
    return reduction_mapping[reduction](similarity)