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
def _check_retrieval_target_and_prediction_types(preds: Tensor, target: Tensor, allow_non_binary_target: bool=False) -> Tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.

    """
    if target.dtype not in (torch.bool, torch.long, torch.int) and (not torch.is_floating_point(target)):
        raise ValueError('`target` must be a tensor of booleans, integers or floats')
    if not preds.is_floating_point():
        raise ValueError('`preds` must be a tensor of floats')
    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError('`target` must contain `binary` values')
    target = target.float() if target.is_floating_point() else target.long()
    preds = preds.float()
    return (preds.flatten(), target.flatten())