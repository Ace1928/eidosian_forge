from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _get_im2col_indices_along_dim(g: jit_utils.GraphContext, input_d, kernel_size_d, dilation_d, padding_d, stride_d):
    blocks_d = g.op('Add', input_d, g.op('Constant', value_t=torch.tensor(padding_d * 2)))
    blocks_d = g.op('Sub', blocks_d, g.op('Constant', value_t=torch.tensor(dilation_d * (kernel_size_d - 1))))
    blocks_d_indices = g.op('Range', g.op('Constant', value_t=torch.tensor(0)), blocks_d, g.op('Constant', value_t=torch.tensor(stride_d)))
    kernel_grid = torch.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op('Constant', value_t=kernel_grid.unsqueeze(0))
    blocks_d_indices = symbolic_helper._unsqueeze_helper(g, blocks_d_indices, [0])
    kernel_mask = symbolic_helper._reshape_helper(g, kernel_grid, g.op('Constant', value_t=torch.tensor([-1, 1])))
    block_mask = g.op('Add', blocks_d_indices, kernel_mask)
    return block_mask