from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _groups_stat_transform(group_stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Transform group statistics by creating a tensor for each statistic."""
    return {'tp': torch.stack([stat[0] for stat in group_stats]), 'fp': torch.stack([stat[1] for stat in group_stats]), 'tn': torch.stack([stat[2] for stat in group_stats]), 'fn': torch.stack([stat[3] for stat in group_stats])}