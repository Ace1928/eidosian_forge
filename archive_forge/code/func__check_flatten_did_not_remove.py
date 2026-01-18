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
def _check_flatten_did_not_remove(original, jit_flattened):
    """torch.jit._flatten removes None. Check if it did so in this case."""

    @_beartype.beartype
    def flatten(x):
        if isinstance(x, (list, tuple)):
            for inner in x:
                yield from flatten(inner)
        elif isinstance(x, dict):
            for inner in x.values():
                yield from flatten(inner)
        else:
            yield x
    flattened_with_none = list(flatten(original))
    num_none = len(flattened_with_none) - len(jit_flattened)
    assert num_none >= 0
    if num_none:
        raise ValueError(f"args contained {num_none} None's after flattening. When exporting a ScriptModule or ScriptFunction, no args may be None because that breaks type propagation.")