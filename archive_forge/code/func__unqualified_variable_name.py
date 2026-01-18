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
def _unqualified_variable_name(qualified_name: str) -> str:
    """
        Parse qualified variable name and return the unqualified version.

        Pure numeric atoms are considered inadequate, so this function will look past them,
        and start from the first non-numeric atom.

        Example:
            >>> _unqualified_variable_name('__main__.Foo.bar')
            'bar'
            >>> _unqualified_variable_name('__main__.Foo.bar.0')
            'bar.0'
        """
    name_atoms = qualified_name.split('.')
    for i, atom in reversed(list(enumerate(name_atoms))):
        if not atom.isnumeric():
            return '.'.join(name_atoms[i:])
    return qualified_name