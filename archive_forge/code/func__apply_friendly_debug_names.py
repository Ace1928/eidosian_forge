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
def _apply_friendly_debug_names(graph, params):
    for n in graph.nodes():
        for v in n.inputs():
            old_name = v.debugName()
            if old_name != str(v.unique()):
                continue
            new_name = f'{n.kind()}_{v.unique()}'
            v.setDebugName(new_name)
            if old_name in params:
                params[new_name] = params.pop(old_name)