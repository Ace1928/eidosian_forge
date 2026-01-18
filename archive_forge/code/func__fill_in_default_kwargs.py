from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _fill_in_default_kwargs(node: torch.fx.Node) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""
    if hasattr(node.target, '_schema'):
        node_schema = node.target._schema
    else:
        node_schema = torch.ops.aten.sym_size.int._schema
    complete_args: List[fx_type_utils.Argument] = []
    complete_kwargs: Dict[str, fx_type_utils.Argument] = {}
    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                complete_kwargs[expected_arg.name] = expected_arg.default_value
    return (complete_args, complete_kwargs)