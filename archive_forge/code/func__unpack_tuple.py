from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _unpack_tuple(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(f"ONNX symbolic expected node type 'prim::TupleConstruct', got '{tuple_node.kind()}'.", tuple_value)
    return tuple(tuple_node.inputs())