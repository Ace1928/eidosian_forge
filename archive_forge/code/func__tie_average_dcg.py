from typing import Optional
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
def _tie_average_dcg(target: Tensor, preds: Tensor, discount_cumsum: Tensor) -> Tensor:
    """Translated version of sklearns `_tie_average_dcg` function.

    Args:
        target: ground truth about each document relevance.
        preds: estimated probabilities of each document to be relevant.
        discount_cumsum: cumulative sum of the discount.

    Returns:
        The cumulative gain of the tied elements.

    """
    _, inv, counts = torch.unique(-preds, return_inverse=True, return_counts=True)
    ranked = torch.zeros_like(counts, dtype=torch.float32)
    ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))
    ranked = ranked / counts
    groups = counts.cumsum(dim=0) - 1
    discount_sums = torch.zeros_like(counts, dtype=torch.float32)
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = discount_cumsum[groups].diff()
    return (ranked * discount_sums).sum()