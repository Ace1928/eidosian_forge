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
def _create_states(self, num_groups: int) -> None:
    default = lambda: torch.zeros(num_groups, dtype=torch.long)
    self.add_state('tp', default(), dist_reduce_fx='sum')
    self.add_state('fp', default(), dist_reduce_fx='sum')
    self.add_state('tn', default(), dist_reduce_fx='sum')
    self.add_state('fn', default(), dist_reduce_fx='sum')