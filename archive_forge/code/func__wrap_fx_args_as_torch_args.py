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
def _wrap_fx_args_as_torch_args(fx_args: List[fx_type_utils.Argument], fx_kwargs: Dict[str, fx_type_utils.Argument], fx_graph_module: torch.fx.GraphModule) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Prepare torch format args and kwargs for op-level validation by using fake tensor to create real tensor to feed in ops"""
    torch_args: List[fx_type_utils.Argument] = _fx_args_to_torch_args(fx_args, fx_graph_module)
    return (torch_args, fx_kwargs)