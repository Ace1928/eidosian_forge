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
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: List[Expr]

    def __str__(self, names=('ranges',)):
        return self.str_helper([f"'{self.device.type}'", str(self.dtype), self.inner_fn_str()] + [f'{name}={getattr(self, name)}' for name in names] + [f'origin_node={self.origin_node!r}'])

    def __post_init__(self):
        super().__post_init__()
        self.origin_node = None
    __repr__ = __str__

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def get_origin_node(self):
        return self.origin_node

    def get_size(self):
        return self.ranges

    def is_extern(self):
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        origin_node = kwargs.pop('origin_node', None)
        tb = kwargs.pop('traceback', None)
        r = cls(*args, **kwargs)
        r.origin_node = origin_node
        r.traceback = tb or traceback.format_stack() if config.debug_ir_traceback else None
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges, prefix='i'):
        return [sympy.Integer(0) if s == 1 else sympy_symbol(f'{prefix}{n}') for n, s in enumerate(ranges)]

    @cache_on_self
    def inner_fn_str_len(self):
        return len(self.inner_fn_str())

    def inner_fn_str(self):
        index = self._index(self.ranges)
        return V.KernelFormatterHandler.ir_to_string(self.inner_fn, index)

    def get_reads(self):
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            if self.get_reduction_type():
                return extract_read_writes(self.make_loader(), self.get_size(), self.get_reduction_size()).reads
            else:
                return extract_read_writes(self.make_loader(), self.get_size()).reads

    def get_reduction_size(self):
        raise NotImplementedError(f'get_reduction_size() is not implemented by {type(self)}!')

    def get_reduction_type(self):
        raise NotImplementedError(f'get_reduction_type() is not implemented by {type(self)}!')

    def constant_to_device(self, device):
        raise NotImplementedError(f'constant_to_device() is not implemented by {type(self)}!')