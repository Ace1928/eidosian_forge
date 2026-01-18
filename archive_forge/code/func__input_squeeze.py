import multiprocessing
import os
import sys
from functools import partial
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, no_type_check
from unittest.mock import Mock
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import select_topk, to_onehot
from torchmetrics.utilities.enums import DataType
def _input_squeeze(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Remove excess dimensions."""
    if preds.shape[0] == 1:
        preds, target = (preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0))
    else:
        preds, target = (preds.squeeze(), target.squeeze())
    return (preds, target)