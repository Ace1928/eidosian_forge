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
def _run_symbolic_method(g, op_name, symbolic_fn, args):
    """
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        graph_context = jit_utils.GraphContext(graph=g, block=g.block(), opset=GLOBALS.export_onnx_opset_version, original_node=None, params_dict=_params_dict, env={})
        return symbolic_fn(graph_context, *args)
    except TypeError as e:
        e.args = (f'{e.args[0]} (occurred when translating {op_name})',)
        raise