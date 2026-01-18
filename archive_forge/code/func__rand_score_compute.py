import torch
from torch import Tensor
from torchmetrics.functional.clustering.utils import (
def _rand_score_compute(contingency: Tensor) -> Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    pair_matrix = calculate_pair_cluster_confusion_matrix(contingency=contingency)
    numerator = pair_matrix.diagonal().sum()
    denominator = pair_matrix.sum()
    if numerator == denominator or denominator == 0:
        return torch.ones_like(numerator, dtype=torch.float32)
    return numerator / denominator