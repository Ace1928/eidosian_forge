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
def _check_top_k(top_k: int, case: str, implied_classes: int, multiclass: Optional[bool], preds_float: bool) -> None:
    if case == DataType.BINARY:
        raise ValueError('You can not use `top_k` parameter with binary data.')
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError('The `top_k` has to be an integer larger than 0.')
    if not preds_float:
        raise ValueError('You have set `top_k`, but you do not have probability predictions.')
    if multiclass is False:
        raise ValueError('If you set `multiclass=False`, you can not set `top_k`.')
    if case == DataType.MULTILABEL and multiclass:
        raise ValueError('If you want to transform multi-label data to 2 class multi-dimensionalmulti-class data using `multiclass=True`, you can not use `top_k`.')
    if top_k >= implied_classes:
        raise ValueError('The `top_k` has to be strictly smaller than the `C` dimension of `preds`.')