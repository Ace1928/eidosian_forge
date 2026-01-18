from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _symbolic_context_handler(symbolic_fn: Callable) -> Callable:
    """Decorator that provides the symbolic context to the symbolic function if needed."""
    if _need_symbolic_context(symbolic_fn):
        warnings.warn('The first argument to symbolic functions is deprecated in 1.13 and will be removed in the future. Please annotate treat the first argument (g) as GraphContext and use context information from the object instead.', category=FutureWarning)

        def wrapper(graph_context: jit_utils.GraphContext, *args, **kwargs):
            symbolic_context = _exporter_states.SymbolicContext(params_dict=graph_context.params_dict, env=graph_context.env, cur_node=graph_context.original_node, onnx_block=graph_context.block)
            return symbolic_fn(symbolic_context, graph_context, *args, **kwargs)
        return wrapper
    return symbolic_fn