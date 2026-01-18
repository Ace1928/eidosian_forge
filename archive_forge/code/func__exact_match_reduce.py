from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
def _exact_match_reduce(correct: Tensor, total: Tensor) -> Tensor:
    """Reduce exact match."""
    return _safe_divide(correct, total)