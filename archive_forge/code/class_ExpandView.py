import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @staticmethod
    def _normalize_size(x, new_size):
        """Replace `-1` with correct sizes"""
        new_size = list(map(sympy.expand, new_size))
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
        return new_size

    @classmethod
    def create(cls, x, new_size):
        new_size = cls._normalize_size(x, new_size)
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, list(new_size), new_stride, old_layout.offset)
            return ReinterpretView(storage, new_layout)
        return ExpandView(x, new_size)

    def get_size(self):
        return self.size

    def make_reindexer(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(index):
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    index[i] = sympy.Integer(0)
            return index
        return reindex