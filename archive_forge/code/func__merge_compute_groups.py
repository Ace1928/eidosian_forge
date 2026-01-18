from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import ModuleDict
from typing_extensions import Literal
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import _flatten_dict, allclose
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
def _merge_compute_groups(self) -> None:
    """Iterate over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one. The complexity of the method is approximately
        ``O(number_of_metrics_in_collection ** 2)``, as all metrics need to be compared to all other metrics.

        """
    num_groups = len(self._groups)
    while True:
        for cg_idx1, cg_members1 in deepcopy(self._groups).items():
            for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                if cg_idx1 == cg_idx2:
                    continue
                metric1 = getattr(self, cg_members1[0])
                metric2 = getattr(self, cg_members2[0])
                if self._equal_metric_states(metric1, metric2):
                    self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                    break
            if len(self._groups) != num_groups:
                break
        if len(self._groups) == num_groups:
            break
        num_groups = len(self._groups)
    temp = deepcopy(self._groups)
    self._groups = {}
    for idx, values in enumerate(temp.values()):
        self._groups[idx] = values