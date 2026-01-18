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
def _init_compute_groups(self) -> None:
    """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group

        """
    if isinstance(self._enable_compute_groups, list):
        self._groups = dict(enumerate(self._enable_compute_groups))
        for v in self._groups.values():
            for metric in v:
                if metric not in self:
                    raise ValueError(f'Input {metric} in `compute_groups` argument does not match a metric in the collection. Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}')
        self._groups_checked = True
    else:
        self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}