from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.group_fairness import (
from torchmetrics.functional.classification.stat_scores import _binary_stat_scores_arg_validation
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class _AbstractGroupStatScores(Metric):
    """Create and update states for computing group stats tp, fp, tn and fn."""
    tp: Tensor
    fp: Tensor
    tn: Tensor
    fn: Tensor

    def _create_states(self, num_groups: int) -> None:
        default = lambda: torch.zeros(num_groups, dtype=torch.long)
        self.add_state('tp', default(), dist_reduce_fx='sum')
        self.add_state('fp', default(), dist_reduce_fx='sum')
        self.add_state('tn', default(), dist_reduce_fx='sum')
        self.add_state('fn', default(), dist_reduce_fx='sum')

    def _update_states(self, group_stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        for group, stats in enumerate(group_stats):
            tp, fp, tn, fn = stats
            self.tp[group] += tp
            self.fp[group] += fp
            self.tn[group] += tn
            self.fn[group] += fn