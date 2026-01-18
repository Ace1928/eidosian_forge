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
def _check_num_classes_mc(preds: Tensor, target: Tensor, num_classes: int, multiclass: Optional[bool], implied_classes: int) -> None:
    """Check consistency of `num_classes`, data and `multiclass` param for (multi-dimensional) multi-class data."""
    if num_classes == 1 and multiclass is not False:
        raise ValueError('You have set `num_classes=1`, but predictions are integers. If you want to convert (multi-dimensional) multi-class data with 2 classes to binary/multi-label, set `multiclass=False`.')
    if num_classes > 1:
        if multiclass is False and implied_classes != num_classes:
            raise ValueError('You have set `multiclass=False`, but the implied number of classes  (from shape of inputs) does not match `num_classes`. If you are trying to transform multi-dim multi-class data with 2 classes to multi-label, `num_classes` should be either None or the product of the size of extra dimensions (...). See Input Types in Metrics documentation.')
        if target.numel() > 0 and num_classes <= target.max():
            raise ValueError('The highest label in `target` should be smaller than `num_classes`.')
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError('The size of C dimension of `preds` does not match `num_classes`.')