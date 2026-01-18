import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppVecOverrides(CppOverrides):
    """Map element-wise ops to aten vectorization C++"""

    def __new__(cls, *args, **kargs):
        self = super().__new__(cls)

        def wrap(func):

            def wrapper(*args, **kwargs):
                has_scalar = any((not arg.is_vec for arg in args if isinstance(arg, CppCSEVariable)))
                has_vector = any((arg.is_vec for arg in args if isinstance(arg, CppCSEVariable)))
                new_args = list(args)
                if has_scalar and has_vector:
                    new_args = []
                    for arg in args:
                        if isinstance(arg, CppCSEVariable) and (not arg.is_vec):
                            assert isinstance(V.kernel, CppVecKernel)
                            new_arg = V.kernel.broadcast(arg)
                            new_args.append(new_arg)
                        else:
                            new_args.append(arg)
                if has_vector:
                    return func(*new_args, **kwargs)
                else:
                    scalar_ops = super(CppVecOverrides, self)
                    scalar_func = getattr(scalar_ops, func.__name__, scalar_ops.__getattr__(func.__name__))
                    assert scalar_func is not None
                    return scalar_func(*args, **kwargs)
            return wrapper
        for name, method in vars(cls).items():
            if getattr(method, '__class__', None) == staticmethod and name != 'masked':
                setattr(self, name, wrap(method.__func__))
        return self

    @staticmethod
    def add(a, b):
        return f'{a} + {b}'

    @staticmethod
    def sub(a, b):
        return f'{a} - {b}'

    @staticmethod
    def mul(a, b):
        return f'{a} * {b}'

    @staticmethod
    def truediv(a, b):
        return f'{a} / {b}'

    @staticmethod
    def abs(x):
        return f'{x}.abs()'

    @staticmethod
    def sin(x):
        return f'{x}.sin()'

    @staticmethod
    def cos(x):
        return f'{x}.cos()'

    @staticmethod
    def exp(x):
        return f'{x}.exp()'

    @staticmethod
    def exp2(x):
        return f'{x}.exp2()'

    @staticmethod
    def expm1(x):
        vec_one = f'decltype({x})(1)'
        return f'{x}.exp() - {vec_one}'

    @staticmethod
    def erf(x):
        return f'{x}.erf()'

    @staticmethod
    def erfc(x):
        return f'{x}.erfc()'

    @staticmethod
    def erfinv(x):
        return f'{x}.erfinv()'

    @staticmethod
    def sqrt(x):
        return f'{x}.sqrt()'

    @staticmethod
    def eq(x, y):
        return f'to_float_mask({x} == {y})'

    @staticmethod
    def ne(x, y):
        return f'to_float_mask({x} != {y})'

    @staticmethod
    def lt(x, y):
        return f'to_float_mask({x} < {y})'

    @staticmethod
    def gt(x, y):
        return f'to_float_mask({x} > {y})'

    @staticmethod
    def le(x, y):
        return f'to_float_mask({x} <= {y})'

    @staticmethod
    def ge(x, y):
        return f'to_float_mask({x} >= {y})'

    @staticmethod
    def and_(x, y):
        return f'{x} & {y}'

    @staticmethod
    def rsqrt(x):
        return f'{x}.rsqrt()'

    @staticmethod
    def pow(a, b):
        return f'{a}.pow({b})'

    @staticmethod
    def log(x):
        return f'{x}.log()'

    @staticmethod
    def round(x):
        return f'{x}.round()'

    @staticmethod
    def floor(x):
        return f'{x}.floor()'

    @staticmethod
    def ceil(x):
        return f'{x}.ceil()'

    @staticmethod
    def trunc(x):
        return f'{x}.trunc()'

    @staticmethod
    def fmod(a, b):
        return f'{a}.fmod({b})'

    @staticmethod
    def lgamma(x):
        return f'{x}.lgamma()'

    @staticmethod
    def logical_and(a, b):
        return f'({a} != 0) & ({b} != 0)'

    @staticmethod
    def logical_not(a):
        return f'{a} == 0'

    @staticmethod
    def logical_or(a, b):
        return f'({a} != 0) | ({b} != 0)'

    @staticmethod
    def logical_xor(a, b):
        return f'({a} != 0) ^ ({b} != 0)'

    @staticmethod
    def tan(a):
        return f'{a}.tan()'

    @staticmethod
    def tanh(a):
        vec_one = f'decltype({a})(1)'
        vec_two = f'decltype({a})(2)'
        vec_minus_two = f'decltype({a})(-2)'
        return f'{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}'

    @staticmethod
    def reciprocal(a):
        return f'{a}.reciprocal()'

    @staticmethod
    def atan(x):
        return f'{x}.atan()'

    @staticmethod
    def acos(x):
        return f'{x}.acos()'

    @staticmethod
    def asin(x):
        return f'{x}.asin()'

    @staticmethod
    def cosh(x):
        return f'{x}.cosh()'

    @staticmethod
    def sinh(x):
        return f'{x}.sinh()'

    @staticmethod
    def log10(x):
        return f'{x}.log10()'

    @staticmethod
    def nextafter(x):
        return f'{x}.nextafter()'

    @staticmethod
    def copysign(a, b):
        return f'{a}.copysign({b})'

    @staticmethod
    def atan2(a, b):
        return f'{a}.atan2({b})'

    @staticmethod
    def hypot(a, b):
        return f'{a}.hypot({b})'

    @staticmethod
    def atanh(x):
        vec_one = f'decltype({x})(1)'
        vec_one_half = f'decltype({x})(0.5)'
        return f'{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()'

    @staticmethod
    def asinh(x):
        vec_one = f'decltype({x})(1)'
        return f'({x} + ({vec_one} + {x}*{x}).sqrt()).log()'

    @staticmethod
    def acosh(x):
        vec_one = f'decltype({x})(1)'
        return f'({x} + ({x}*{x} - {vec_one}).sqrt()).log()'

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'{x}; throw 1'
        elif bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'at::vec::clamp_min({x}, decltype({x})(0))'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def sigmoid(x):
        return f'decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())'

    @staticmethod
    def neg(x):
        return f'{x}.neg()'

    @staticmethod
    def floordiv(a, b):
        _t = f'decltype({a})'
        quot = f'{a} / {b}'
        rem = f'{a} % {b}'
        return f'(({a} < {_t}(0)) != ({b} < {_t}(0)) ? ({rem} != {_t}(0) ? {quot} - {_t}(1) : {quot}) : {quot})'

    @staticmethod
    def truncdiv(a, b):
        return f'{a} / {b}'

    @staticmethod
    def minimum(a, b):
        return f'at::vec::minimum({a}, {b})'

    @staticmethod
    def maximum(a, b):
        return f'at::vec::maximum({a}, {b})'

    @staticmethod
    def square(a):
        return f'{a} * {a}'

    @staticmethod
    def where(a, b, c):
        return f'decltype({b})::blendv({c}, {b}, {a})'

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        vec_zero = f'decltype({x})(0)'
        vec_one = f'decltype({x})(1)'
        blendv = f'decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})'
        left = V.kernel.cse.newvar()
        code.writeline(f'auto {left} = {blendv};')
        blendv = f'decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})'
        right = V.kernel.cse.newvar()
        code.writeline(f'auto {right} = {blendv};')
        result = V.kernel.cse.newvar()
        code.writeline(f'auto {result} = {left} - {right};')
        V.kernel.compute.splice(code)
        return result

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        assert dtype in [torch.bool, torch.float, torch.bfloat16, torch.float16, torch.uint8], f'{__name__} does not support {dtype}'
        node: torch.fx.Node = V.interpreter.current_node
        assert node and isinstance(node, torch.fx.Node)
        opt_ctx_x = get_opt_ctx(node.args[1])
        assert opt_ctx_x
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype == torch.bool:
            return f'vec_convert_to_mask({x})'
        if opt_ctx_x.dtype == torch.bool and dtype in (torch.float, torch.float32):
            return f'mask_convert_to_float({x})'
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype in DTYPE_LOWP_FP:
            return f'cvt_fp32_to_lowp_fp<{DTYPE_TO_CPP[dtype]}>({x})'
        if opt_ctx_x.dtype in DTYPE_LOWP_FP and dtype in (torch.float, torch.float32):
            return f'cvt_lowp_fp_to_fp32<{DTYPE_TO_CPP[opt_ctx_x.dtype]}>({x})'
        if opt_ctx_x.dtype == torch.uint8 and dtype in (torch.float, torch.float32):
            return f'at::vec::convert_uint8_to_float({x})'
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype == torch.uint8:
            return f'at::vec::convert_float_to_uint8({x})'
        return f'({x})'

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'{x}.log1p()'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        with V.kernel.masked(mask) as new_mask:
            code.writeline(f'auto {var} = [&]')
            with V.kernel.swap_buffers(code), code.indent():
                result = body()
                code.writeline(f'return {result};')
        code.writeline(';')
        V.kernel.compute.splice(code)
        other_code = value_to_cpp(other, 'float')
        other_code_vec = f'at::vec::Vectorized<float>({other_code})'
        if result.is_vec:
            type = f'decltype({var}())'
            float_mask = f'to_float_mask({new_mask})'
            csevar = V.kernel.cse.generate(V.kernel.compute, f'{type}::blendv({other_code_vec}, {var}(), {float_mask})')
        else:
            csevar = V.kernel.cse.generate(V.kernel.compute, f'{mask} ? {var}() : {other_code}')
        csevar.update_on_args('masked', (mask, body, other, result), {})
        return csevar