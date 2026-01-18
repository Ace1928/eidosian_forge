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
def _track_module_attributes_forward_hook(module, input, output):
    tracing_state = _C._get_tracing_state()
    if not tracing_state:
        return
    graph = tracing_state.graph()
    onnx_attrs = {}
    if hasattr(module, attr_name):
        onnx_attrs = getattr(module, attr_name)
        delattr(module, attr_name)
    _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)