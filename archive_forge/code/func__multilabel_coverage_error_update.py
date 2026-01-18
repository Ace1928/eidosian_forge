from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
def _multilabel_coverage_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Accumulate state for coverage error."""
    offset = torch.zeros_like(preds)
    offset[target == 0] = preds.min().abs() + 10
    preds_mod = preds + offset
    preds_min = preds_mod.min(dim=1)[0]
    coverage = (preds >= preds_min[:, None]).sum(dim=1).to(torch.float32)
    return (coverage.sum(), coverage.numel())