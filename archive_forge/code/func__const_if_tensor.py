from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def _const_if_tensor(graph_context: GraphContext, arg):
    if arg is None:
        return arg
    if isinstance(arg, _C.Value):
        return arg
    return _add_op(graph_context, 'onnx::Constant', value_z=arg)