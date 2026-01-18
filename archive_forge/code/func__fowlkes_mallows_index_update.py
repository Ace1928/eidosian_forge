from typing import Tuple
import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels
def _fowlkes_mallows_index_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Return contingency matrix required to compute the Fowlkes-Mallows index.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return (calculate_contingency_matrix(preds, target), preds.size(0))