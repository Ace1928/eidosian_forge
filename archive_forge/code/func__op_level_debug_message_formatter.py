from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _op_level_debug_message_formatter(fn: Callable, self, node: torch.fx.Node, symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction], *args, **kwargs) -> str:
    return f'FX Node: {node.op}::{node.target}[name={node.name}]. \nONNX Node: {symbolic_fn.name}[opset={symbolic_fn.opset}].'