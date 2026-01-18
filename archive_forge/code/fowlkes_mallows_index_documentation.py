from typing import Tuple
import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels
Compute Fowlkes-Mallows index between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with Fowlkes-Mallows index

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering import fowlkes_mallows_index
        >>> preds = torch.tensor([2, 2, 0, 1, 0])
        >>> target = torch.tensor([2, 2, 1, 1, 0])
        >>> fowlkes_mallows_index(preds, target)
        tensor(0.5000)

    