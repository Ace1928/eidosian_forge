import builtins
import copy
import dataclasses
import inspect
import io
import math
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils._pytree import (
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        if min_ == 2:
            min_ = None
        if max_ == sys.maxsize - 1:
            max_ = None
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"