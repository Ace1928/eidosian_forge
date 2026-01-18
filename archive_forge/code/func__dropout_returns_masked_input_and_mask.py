from __future__ import annotations
import functools
import sys
from typing import Optional, Tuple
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _dropout_returns_masked_input_and_mask(g: jit_utils.GraphContext, input: torch._C.Value, p: float, train: bool) -> Tuple[torch._C.Value, Optional[torch._C.Value]]:
    symbolic_helper.check_training_mode(train, 'dropout')
    if not train:
        return (input, None)
    p = g.op('Constant', value_t=torch.tensor(p))
    t = g.op('Constant', value_t=torch.tensor(train, dtype=torch.bool))
    r, mask = g.op('Dropout', input, p, t, outputs=2)
    return (r, mask)