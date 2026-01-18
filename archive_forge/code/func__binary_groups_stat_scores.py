from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _binary_groups_stat_scores(preds: torch.Tensor, target: torch.Tensor, groups: torch.Tensor, num_groups: int, threshold: float=0.5, ignore_index: Optional[int]=None, validate_args: bool=True) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, 'global', ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, 'global', ignore_index)
        _groups_validation(groups, num_groups)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    groups = _groups_format(groups)
    indexes, indices = torch.sort(groups.squeeze(1))
    preds = preds[indices]
    target = target[indices]
    split_sizes = _flexible_bincount(indexes).detach().cpu().tolist()
    group_preds = list(torch.split(preds, split_sizes, dim=0))
    group_target = list(torch.split(target, split_sizes, dim=0))
    return [_binary_stat_scores_update(group_p, group_t) for group_p, group_t in zip(group_preds, group_target)]