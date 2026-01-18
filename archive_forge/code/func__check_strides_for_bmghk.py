import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _check_strides_for_bmghk(x: torch.Tensor, name: str, reasons: List[str]) -> None:
    """
    We want to be able to collapse the G/H dimensions together
    """
    if x.ndim == 5:
        stride_g, stride_h = (x.stride(2), x.stride(3))
        if x.shape[2] == 1:
            return
        if x.shape[3] == 1 or stride_h == 0:
            return
        if stride_g != stride_h * x.shape[-2]:
            reasons.append(f'GQA is only supported when the G/H dimensions are contiguous\n    {name}.stride:  {x.stride()}\n    {name}.shape :  {list(x.shape)}')