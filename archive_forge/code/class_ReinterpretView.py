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
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""
    layout: 'Layout'

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, BaseView):
            self.data = self.data.unwrap_view()

    def __str__(self):
        return self.str_helper([self.data, self.layout])
    __repr__ = __str__

    def get_name(self):
        return self.data.get_name()

    def get_device(self):
        return self.layout.device

    def get_origin_node(self):
        return None

    def get_dtype(self):
        return self.layout.dtype

    def get_size(self):
        return list(self.layout.size)

    def get_stride(self):
        return list(self.layout.stride)

    def make_loader(self):

        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(self.get_name(), indexer(index))
        return loader

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        pass

    def codegen_reference(self, writer=None):
        return V.graph.wrapper_code.codegen_reinterpret_view(self.data, self.layout.size, self.layout.stride, self.layout.offset, writer)