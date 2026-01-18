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
def _basic_input_validation(preds: Tensor, target: Tensor, threshold: float, multiclass: Optional[bool], ignore_index: Optional[int]) -> None:
    """Perform basic validation of inputs that does not require deducing any information of the type of inputs."""
    if _check_for_empty_tensors(preds, target):
        return
    if target.is_floating_point():
        raise ValueError('The `target` has to be an integer tensor.')
    if ignore_index is None and target.min() < 0 or (ignore_index and ignore_index >= 0 and (target.min() < 0)):
        raise ValueError('The `target` has to be a non-negative tensor.')
    preds_float = preds.is_floating_point()
    if not preds_float and preds.min() < 0:
        raise ValueError('If `preds` are integers, they have to be non-negative.')
    if not preds.shape[0] == target.shape[0]:
        raise ValueError('The `preds` and `target` should have the same first dimension.')
    if multiclass is False and target.max() > 1:
        raise ValueError('If you set `multiclass=False`, then `target` should not exceed 1.')
    if multiclass is False and (not preds_float) and (preds.max() > 1):
        raise ValueError('If you set `multiclass=False` and `preds` are integers, then `preds` should not exceed 1.')