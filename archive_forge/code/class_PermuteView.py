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
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        dims = cls._map_neg_dims(dims)
        assert set(dims) == set(range(len(dims)))
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, [old_layout.size[i] for i in dims], [old_layout.stride[i] for i in dims], old_layout.offset)
            return ReinterpretView(storage, new_layout)
        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        assert set(self._map_neg_dims(self.dims)) == set(range(len(self.dims)))
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert set(inv) == set(range(len(self.dims)))

        def reindex(index):
            return [index[i] for i in inv]
        return reindex