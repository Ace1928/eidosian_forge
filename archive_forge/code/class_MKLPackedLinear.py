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
class MKLPackedLinear(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkl._mkl_linear', cpp_kernel='mkl::_mkl_linear')
        self.cpp_kernel_key = 'mkl_linear'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& self,\n                const at::Tensor& mkl_weight_t,\n                const at::Tensor& origin_weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                const int64_t prepack_batch_size)'

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.codegen_kernel_name(), self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)

    @classmethod
    def create(cls, x, packed_w, orig_w, batch_size):
        x = cls.require_stride1(cls.realize_input(x))
        orig_w = cls.require_stride1(cls.realize_input(orig_w))
        *m, _ = x.get_size()
        oc, _ = orig_w.get_size()
        output_size = list(m) + [oc]
        output_stride = make_contiguous_strides_for(output_size)
        inputs = [x, packed_w, orig_w]
        constant_args = [None, batch_size]
        return MKLPackedLinear(layout=FixedLayout(x.get_device(), x.get_dtype(), output_size, output_stride), inputs=inputs, constant_args=constant_args)