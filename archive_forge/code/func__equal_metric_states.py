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
@staticmethod
def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
    """Check if the metric state of two metrics are the same."""
    if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
        return False
    if metric1._defaults.keys() != metric2._defaults.keys():
        return False
    for key in metric1._defaults:
        state1 = getattr(metric1, key)
        state2 = getattr(metric2, key)
        if type(state1) != type(state2):
            return False
        if isinstance(state1, Tensor) and isinstance(state2, Tensor):
            return state1.shape == state2.shape and allclose(state1, state2)
        if isinstance(state1, list) and isinstance(state2, list):
            return all((s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2)))
    return True