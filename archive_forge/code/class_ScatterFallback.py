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
class ScatterFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly.
    This class handles both aten.scatter_ and aten.scatter_reduce_.
    It also handle the case `src` being a scalar properly.
    """

    def codegen(self, wrapper):
        reduce = self.kwargs['reduce']
        if V.graph.cpp_wrapper:
            get_operator_enum = {'add': 'sum', 'multiply': 'prod'}
            if reduce in get_operator_enum:
                reduce = get_operator_enum[reduce]
            self.cpp_kernel = self.get_cpp_kernel(self.fn, reduce)
        if self.src_is_tensor:
            x, index, src = (t.codegen_reference() for t in self.inputs)
        else:
            x, index = (t.codegen_reference() for t in self.inputs)
            src = self.constant_args[1]
        wrapper.generate_scatter_fallback(x, [x, self.constant_args[0], index, src], self.cpp_kernel if V.graph.cpp_wrapper else self.kernel, self.fn, self.src_is_tensor, reduce, self.codegen_kwargs())

    def should_allocate(self):
        return False

    def get_cpp_kernel(self, fn, reduce):
        if fn == 'aten.scatter_':
            if self.src_is_tensor:
                kernel = 'at::scatter_out' if reduce is None else 'at::scatter_reduce_out'
            else:
                assert reduce is None, 'Expect reduce to be None for aten.scatter_ with scalar src'
                kernel = 'at::scatter_out'
        else:
            assert reduce is not None, 'Expect reduce to be not None for aten.scatter_reduce_'
            kernel = 'at::scatter_reduce_out'
        return kernel

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        return {}

    def __init__(self, fn, x, dim: int, index, src, *, reduce: Optional[str]=None, include_self: bool=True):
        assert fn in {'aten.scatter_', 'aten.scatter_reduce_'}
        self.src_is_tensor = isinstance(src, TensorBox)
        self.kernel = fn
        self.fn = fn
        constant_args: Tuple[Any, ...]
        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)
        super().__init__(None, NoneLayout(x.get_device()), self.unwrap_storage(tensors), constant_args, {'reduce': reduce, 'include_self': include_self})
        self.ordered_kwargs_for_cpp_kernel = ['reduce', 'include_self']
        self.name = V.graph.register_buffer(self)
        mark_node_as_mutating(self, x)