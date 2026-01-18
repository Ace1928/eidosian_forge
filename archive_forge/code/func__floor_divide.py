from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::_floor_divide')
@_beartype.beartype
def _floor_divide(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = opset9.true_divide(g, self, other)
        return g.op('Floor', out)
    else:
        div = g.op('Div', self, other)
        zero = g.op('Constant', value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op('Xor', g.op('Less', self, zero), g.op('Less', other, zero))
        mod = g.op('Mod', self, other, fmod_i=0)
        fixup_mask = g.op('And', negative, g.op('Not', g.op('Equal', mod, zero)))
        one = g.op('Constant', value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op('Sub', div, one)
        return g.op('Where', fixup_mask, fixup, div)