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
def _check_num_classes_binary(num_classes: int, multiclass: Optional[bool]) -> None:
    """Check that the consistency of `num_classes` with the data and `multiclass` param for binary data."""
    if num_classes > 2:
        raise ValueError('Your data is binary, but `num_classes` is larger than 2.')
    if num_classes == 2 and (not multiclass):
        raise ValueError('Your data is binary and `num_classes=2`, but `multiclass` is not True. Set it to True if you want to transform binary data to multi-class format.')
    if num_classes == 1 and multiclass:
        raise ValueError('You have binary data and have set `multiclass=True`, but `num_classes` is 1. Either set `multiclass=None`(default) or set `num_classes=2` to transform binary data to multi-class format.')