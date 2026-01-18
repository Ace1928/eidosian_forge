import torch
from torch import Tensor
from torchmetrics.functional.clustering.utils import (
Compute the Adjusted Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with adjusted rand score

    Example:
        >>> from torchmetrics.functional.clustering import adjusted_rand_score
        >>> import torch
        >>> adjusted_rand_score(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
        tensor(1.)
        >>> adjusted_rand_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.5714)

    