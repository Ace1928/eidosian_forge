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
class ExternKernelAlloc(ExternKernel):

    def codegen_kernel_name(self):
        return self.cpp_kernel if V.graph.cpp_wrapper else self.kernel

    def codegen(self, wrapper):
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        V.graph.wrapper_code.generate_extern_kernel_alloc(self, args)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def __init__(self, layout, inputs, constant_args=(), kwargs=None, kernel=None, cpp_kernel=None, ordered_kwargs_for_cpp_kernel=()):
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args, kwargs or {})
        self.name = V.graph.register_buffer(self)
        self.kernel = kernel
        self.cpp_kernel = cpp_kernel
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel

    def should_allocate(self):
        return False

    def apply_constraint(self):
        raise NotImplementedError