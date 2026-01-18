from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _groups_reduce(group_stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Compute rates for all the group statistics."""
    return {f'group_{group}': torch.stack(stats) / torch.stack(stats).sum() for group, stats in enumerate(group_stats)}