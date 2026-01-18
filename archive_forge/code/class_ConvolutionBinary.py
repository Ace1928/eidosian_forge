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
class ConvolutionBinary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=(), cpp_constant_args=()):
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._convolution_pointwise.binary', cpp_kernel='mkldnn::_convolution_pointwise')
        self.cpp_kernel_overload_name = 'binary'
        self.cpp_kernel_key = 'convolution_pointwise_binary'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& other_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                at::IntArrayRef padding,\n                at::IntArrayRef stride,\n                at::IntArrayRef dilation,\n                int64_t groups,\n                c10::string_view binary_attr,\n                c10::optional<at::Scalar> alpha,\n                c10::optional<c10::string_view> unary_attr,\n                torch::List<c10::optional<at::Scalar>> unary_scalars,\n                c10::optional<c10::string_view> unary_algorithm)'
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.codegen_kernel_name(), self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key, self.cpp_kernel_overload_name)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x: 'TensorBox', other: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding_: List[int], stride_: List[int], dilation_: List[int], groups: int, binary_attr: str, binary_alpha: Optional[float], unary_attr: Optional[str], unary_scalars: Optional[List[Any]], unary_algorithm: Optional[str]):
        inputs, constant_args, kernel_layout, req_stride_order = _prepare_convolution_fusion_create(cls, x, weight, bias, padding_, stride_, dilation_, groups)
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [binary_attr, binary_alpha, unary_attr, may_convert_to_optional(unary_scalars), unary_algorithm]
        return ConvolutionBinary(layout=kernel_layout, inputs=inputs, constant_args=constant_args)