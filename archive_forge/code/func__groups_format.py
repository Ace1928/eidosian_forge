from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _groups_format(groups: torch.Tensor) -> torch.Tensor:
    """Reshape groups to correspond to preds and target."""
    return groups.reshape(groups.shape[0], -1)