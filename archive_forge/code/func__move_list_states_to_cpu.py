import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _move_list_states_to_cpu(self) -> None:
    """Move list states to cpu to save GPU memory."""
    for key in self._defaults:
        current_val = getattr(self, key)
        current_to_cpu = []
        if isinstance(current_val, Sequence):
            for cur_v in current_val:
                if not isinstance(cur_v, tuple):
                    cur_v = cur_v.to('cpu')
                current_to_cpu.append(cur_v)
        setattr(self, key, current_to_cpu)