from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype]=None):

        def _get_min_elements_per_thread(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> int:
            if src_dtype == dst_dtype:
                return 0
            fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
            assert not (src_dtype in fp8_dtypes and dst_dtype in fp8_dtypes and (src_dtype != dst_dtype)), 'Conversions between float8_e5m2 and float8_e4m3fn is not supported!'
            if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
                return 4
            if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
                return 2
            return 0
        if src_dtype is not None:
            V.kernel.min_elem_per_thread = max(_get_min_elements_per_thread(src_dtype, dtype), V.kernel.min_elem_per_thread)
        if dtype == torch.bool:
            return f'({x} != 0)'
        elif dtype == torch.uint8:
            return f'{x}.to(tl.int8).to(tl.uint8)'
        return f'{x}.to({triton_compute_type(dtype)})'

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype):
        return f'{x}.to({triton_compute_type(dtype)}, bitcast=True)'

    @classmethod
    def constant(cls, value, dtype):
        if dtype == torch.uint8:
            tmp = cls.constant(value, torch.int16)
            return cls.to_dtype(tmp, dtype)
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = triton_constant(type_(value))
        triton_type = triton_compute_type(dtype)
        if triton_type == 'tl.float32':
            return triton_val
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return f'tl.full({shape}, {triton_val}, {triton_type})'

    @staticmethod
    def abs(x):
        return f'tl.abs({x})'

    @staticmethod
    def libdevice_abs(x):
        return f'tl.math.abs({x})'

    @staticmethod
    def exp(x):
        return f'tl.exp({x})'

    @staticmethod
    def libdevice_exp(x):
        return f'tl.math.exp({x})'

    @staticmethod
    def exp2(x):
        return f'tl.math.exp2({x})'

    @staticmethod
    def expm1(x):
        return f'tl.math.expm1({x})'

    @staticmethod
    def sqrt(x):
        return f'tl.sqrt({x})'

    @staticmethod
    def libdevice_sqrt(x):
        return f'tl.math.sqrt({x})'

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == 'accuracy':
            return f'{x} + 1'
        elif bug is None:
            return ops.maximum('0', x)
        else:
            raise AssertionError(f'unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def minimum(a, b):
        return f'triton_helpers.minimum({a}, {b})'

    @staticmethod
    def maximum(a, b):
        return f'triton_helpers.maximum({a}, {b})'

    @staticmethod
    def where(a, b, c):
        return f'tl.where({a}, {b}, {c})'

    @staticmethod
    def cos(x):
        return f'tl.cos({x})'

    @staticmethod
    def libdevice_cos(x):
        return f'tl.math.cos({x})'

    @staticmethod
    def sin(x):
        return f'tl.sin({x})'

    @staticmethod
    def libdevice_sin(x):
        return f'tl.math.sin({x})'

    @classmethod
    def index_expr(cls, expr, dtype):
        index_str, mask_vars, mask, expand_str = V.kernel.indexing(expr)
        var = V.kernel.cse.generate(V.kernel.compute, index_str)
        if dtype not in {torch.int32, torch.int64}:
            var = V.kernel.cse.generate(V.kernel.compute, cls.to_dtype(var, dtype))
        var.mask_vars = mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        other = V.kernel.cse.generate(V.kernel.compute, f'tl.full({result}.shape, {triton_constant(other)}, {result}.dtype)')
        return ops.where(new_mask, result, other)

    @staticmethod
    def lgamma(x):
        return f'tl.math.lgamma({x})'

    @staticmethod
    def erf(x):
        return f'tl.math.erf({x})'

    @staticmethod
    def cosh(x):
        return f'tl.math.cosh({x})'

    @staticmethod
    def sinh(x):
        return f'tl.math.sinh({x})'

    @staticmethod
    def acos(x):
        return f'tl.math.acos({x})'

    @staticmethod
    def acosh(x):
        return f'tl.math.acosh({x})'

    @staticmethod
    def asin(x):
        return f'tl.math.asin({x})'

    @staticmethod
    def asinh(x):
        return f'tl.math.asinh({x})'

    @staticmethod
    def atan2(x, y):
        return f'tl.math.atan2({x}, {y})'

    @staticmethod
    def atan(x):
        return f'tl.math.atan({x})'

    @staticmethod
    def atanh(x):
        return f'tl.math.atanh({x})'

    @staticmethod
    def copysign(x, y):
        return f'tl.math.copysign({x}, {y})'

    @staticmethod
    def erfc(x):
        return f'tl.math.erfc({x})'

    @staticmethod
    def erfinv(x):
        return f'tl.math.erfinv({x})'

    @staticmethod
    def hypot(x, y):
        return f'tl.math.hypot({x}, {y})'

    @staticmethod
    def log10(x):
        return f'tl.math.log10({x})'

    @staticmethod
    def nextafter(x, y):
        return f'tl.math.nextafter({x}, {y})'

    @staticmethod
    def logical_and(a, b):
        return f'{a} & {b}'

    @staticmethod
    def logical_not(a):
        return f'{a} == 0'

    @staticmethod
    def logical_or(a, b):
        return f'{a} | {b}'

    @staticmethod
    def logical_xor(a, b):
        return f'({a} ^ {b})'

    @staticmethod
    def bitwise_and(a, b):
        return f'{a} & {b}'

    @staticmethod
    def bitwise_not(a):
        return f'~{a}'

    @staticmethod
    def bitwise_or(a, b):
        return f'{a} | {b}'

    @staticmethod
    def bitwise_xor(a, b):
        return f'{a} ^ {b}'

    @staticmethod
    def bitwise_left_shift(a, b):
        return f'{a} << {b}'

    @staticmethod
    def bitwise_right_shift(a, b):
        return f'{a} >> {b}'

    @staticmethod
    def rand(seed, offset):
        offset = f'({offset}).to(tl.uint32)'
        return f'tl.rand({seed}, {offset})'

    @staticmethod
    def randn(seed, offset):
        offset = f'({offset}).to(tl.uint32)'
        return f'tl.randn({seed}, {offset})'

    @staticmethod
    def randint64(seed, offset, low, high):
        offset = f'({offset}).to(tl.uint32)'
        return f'triton_helpers.randint64({seed}, {offset}, {low}, {high})'

    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return f'tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})'

    @staticmethod
    def rsqrt(x):
        return f'tl.math.rsqrt({x})'

    @staticmethod
    def log1p(x):
        return f'tl.math.log1p({x})'

    @staticmethod
    def tan(x):
        return f'tl.math.tan({x})'

    @staticmethod
    def tanh(x):
        return f'tl.math.tanh({x})'

    @staticmethod
    def sigmoid(x):
        return f'tl.sigmoid({x})'

    @staticmethod
    def libdevice_sigmoid(x):
        return f'1/(1 + tl.math.exp(-({x})))'

    @staticmethod
    def signbit(x):
        return f'tl.math.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0'

    @staticmethod
    def fmod(a, b):
        return f'tl.math.fmod({a}, {b})'

    @staticmethod
    def pow(a, b):
        return f'tl.math.pow({a}, {b})'

    @staticmethod
    def log(x):
        return f'tl.log({x})'

    @staticmethod
    def libdevice_log(x):
        return f'tl.math.log({x})'

    @staticmethod
    def isinf(x):
        return f'tl.math.isinf({x}).to(tl.int1)'

    @staticmethod
    def isnan(x):
        return f'tl.math.isnan({x}).to(tl.int1)'

    @staticmethod
    def round(x):
        return f'tl.math.nearbyint({x})'

    @staticmethod
    def floor(x):
        return f'tl.math.floor({x})'

    @staticmethod
    def floordiv(a, b):
        quot = f'{a} // {b}'
        rem = f'{a} % {b}'
        return f'tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})'

    @staticmethod
    def sign(x):

        def to_int(s):
            return f'{s}.to(tl.int8)'
        left = to_int(ops.lt('0', x))
        right = to_int(ops.lt(x, '0'))
        sub = ops.sub(left, right)
        return f'{sub}.to({x}.dtype)'

    @staticmethod
    def trunc(x):
        return f'tl.math.trunc({x})'

    @staticmethod
    def truncdiv(a, b):
        return f'{a} // {b}'

    @staticmethod
    def ceil(x):
        return f'tl.math.ceil({x})'