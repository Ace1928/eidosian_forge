from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide
def _critical_success_index_compute(hits: Tensor, misses: Tensor, false_alarms: Tensor) -> Tensor:
    """Compute critical success index.

    Args:
        hits: Number of true positives after binarization
        misses: Number of false negatives after binarization
        false_alarms: Number of false positives after binarization

    Returns:
        If input tensors are 5-dimensional and ``keep_sequence_dim=True``, the metric returns a ``(S,)`` vector
        with CSI scores for each image in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    """
    return _safe_divide(hits, hits + misses + false_alarms)