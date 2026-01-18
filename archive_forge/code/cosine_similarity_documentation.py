from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
Compute the `Cosine Similarity`_.

    .. math::
        cos_{sim}(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||} =
        \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    Args:
        preds: Predicted tensor with shape ``(N,d)``
        target: Ground truth tensor with shape ``(N,d)``
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> from torchmetrics.functional.regression import cosine_similarity
        >>> target = torch.tensor([[1, 2, 3, 4],
        ...                        [1, 2, 3, 4]])
        >>> preds = torch.tensor([[1, 2, 3, 4],
        ...                       [-1, -2, -3, -4]])
        >>> cosine_similarity(preds, target, 'none')
        tensor([ 1.0000, -1.0000])

    