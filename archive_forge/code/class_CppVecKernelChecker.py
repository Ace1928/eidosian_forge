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
class CppVecKernelChecker(CppVecKernel):

    def __init__(self, args, num_threads, tiling_factor, tiling_idx=-1):
        super().__init__(args, num_threads, tiling_factor, tiling_idx)
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1
        self._orig_wrapper_code = None
        self.simd_vec = True
        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()
        self.load_supported_dtypes: List[torch.dtype] = [torch.float, torch.bfloat16, torch.float16, torch.bool, torch.uint8]
        self.store_supported_dtypes: List[torch.dtype] = [torch.float, torch.bfloat16, torch.float16, torch.uint8]
        self.store_dtypes: List[torch.dtype] = []
        self.vec_dtype: torch.dtype = torch.float32

    def disable_vec(self, msg=None):
        if schedule_log.isEnabledFor(logging.DEBUG):
            schedule_log.debug('Disabled vectorization: %s', msg)
        self.simd_vec = False

    def is_mask(self, name: str, users: Dict[torch.fx.Node, None]):
        load_type = V.graph.get_dtype(name)
        if load_type == torch.bool:
            return all((user.target in ('where', 'masked') for user in users.keys()))
        elif load_type == torch.uint8:
            '\n            If the load value is torch.uint8, then we only support the loaded\n            value is as the mask.\n            '
            if not all((user.target == 'to_dtype' and user.args[-1] == torch.bool for user in users.keys())):
                return False
            for to_dtype_node in users.keys():
                assert to_dtype_node.target == 'to_dtype'
                if not all((user.target in ('where', 'masked') for user in to_dtype_node.users.keys())):
                    return False
            return True
        else:
            return False

    def is_load_uint8_as_float(self, name: str, users: Dict[torch.fx.Node, None]):
        """
        Check:
        1. load_type is torch.uint8
        2. has 1 user node of target to_dtype
        3. dtype of to_dtype is torch.float
        """
        load_type = V.graph.get_dtype(name)
        if load_type is not torch.uint8:
            return False
        if len(users) == 1:
            user = next(iter(users))
            if user.target == 'to_dtype' and user.args[-1] == torch.float:
                return True
            return False
        return False

    def can_store_fp32_as_uint8(self, store_var: str, value_node: torch.fx.Node):
        """
        Check:
        1. store_type is torch.uint8
        2. value_node is of target to_dtype
        3. dtype of to_dtype node is torch.uint8
        """
        store_type = V.graph.get_dtype(store_var)
        if store_type not in [torch.uint8]:
            return False
        if value_node.target == 'to_dtype' and value_node.args[-1] == torch.uint8:
            return True
        return False

    def is_load_integer_scalar_tensor(self, name: str, index: sympy.Expr):
        load_dtype = V.graph.get_dtype(name)
        buffer = V.graph.get_buffer(name)
        return load_dtype in [torch.int32, torch.int64] and isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0) and (index == 0)

    def load(self, name: str, index: sympy.Expr):
        with RecordOptimizationContext(__name__) as node_ctx:
            load_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = load_dtype
            opt_ctx.is_load_as_mask = self.is_mask(name, node_ctx.get_fx_node().users)
            opt_ctx.is_load_uint8_as_float = self.is_load_uint8_as_float(name, node_ctx.get_fx_node().users)
            var = self.cse.newvar()
            if len(self.itervars) == 0:
                self.disable_vec('not a loop')
                return var
            if load_dtype in [torch.bool, torch.uint8] and (not (opt_ctx.is_load_as_mask or opt_ctx.is_load_uint8_as_float)):
                if not opt_ctx.is_load_as_mask:
                    self.disable_vec(f'{load_dtype} not loaded as mask')
                elif not opt_ctx.is_load_uint8_as_float:
                    self.disable_vec(f'{load_dtype} not loaded as float')
                return var
            if load_dtype not in self.load_supported_dtypes and (not self.is_load_integer_scalar_tensor(name, index)) and index.has(self.itervars[self.tiling_idx]):
                self.disable_vec(f'{load_dtype} not supported by load')
                return var
            return var

    def store(self, name, index, value, mode=None):
        with RecordOptimizationContext(__name__) as node_ctx:
            if len(self.itervars) == 0:
                self.disable_vec('not a loop')
                return self.simd_vec
            store_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = store_dtype
            store_dtype = torch.float if store_dtype == torch.float32 else store_dtype
            self.store_dtypes.append(store_dtype)
            if store_dtype not in self.store_supported_dtypes:
                self.disable_vec(f'{store_dtype} not supported by store')
                return self.simd_vec
            if store_dtype in [torch.uint8]:
                value_node = node_ctx.get_fx_node().all_input_nodes[-1]
                if not self.can_store_fp32_as_uint8(name, value_node):
                    self.disable_vec('not support store float32 as uint8')
                    return self.simd_vec
            assert 'buf' in name
            index = self.rename_indexing(index)
            if mode:
                self.disable_vec(f'store mode: {mode}')
                return self.simd_vec
            if index.is_number:
                self.disable_vec(f'constant store index: {index}')
            return self.simd_vec

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if dtype == torch.float and src_dtype == torch.float and (reduction_type in VECTORIZABLE_RTYPES):
            pass
        else:
            self.disable_vec(f'reduction: dtype {dtype}, src_dtype {src_dtype}, reduction_type {reduction_type}')
        if is_welford_reduction(reduction_type):
            return tuple([self.simd_vec] * 3)
        return self.simd_vec

    def store_reduction(self, name, index, value):
        return self.simd_vec

    def is_supported_cmp(self, node: torch.fx.Node):

        def get_node_dtype(node):
            if type(node) == torch.fx.Node:
                opt_ctx: OptimizationContext = get_current_node_opt_ctx()
                return opt_ctx.dtype if opt_ctx else None
            else:
                return None

        def get_cmp_dtypes(node: torch.fx.Node):
            return (get_node_dtype(node.args[-2]), get_node_dtype(node.args[-1]))
        assert len(node.args) >= 2
        if type(node.args[-1]) in [int, float]:
            return True
        if type(node.args[-2]) in [int, float]:
            return False
        left_dtype, right_dtype = get_cmp_dtypes(node)
        if left_dtype is None or right_dtype is None:
            return True
        else:
            return left_dtype == right_dtype

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._orig_wrapper_code is not None
        V.graph.wrapper_code = self._orig_wrapper_code
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        class VecCheckerProxy:
            bin_cmp_ops = ['eq', 'ne', 'le', 'ge', 'lt', 'gt']

            @staticmethod
            def _bin_cmp_op(x, y):
                current_node: torch.fx.Node = V.interpreter.current_node
                if not self.is_supported_cmp(current_node):
                    self.disable_vec(f'binary comparison op: {current_node}')
                return self.simd_vec

            @staticmethod
            def __getattr__(name):

                def inner(*args, **kwargs):
                    if name in VecCheckerProxy.bin_cmp_ops:
                        return VecCheckerProxy._bin_cmp_op(args, kwargs)
                    if name not in self.fast_vec_list:
                        self.disable_vec(f'op: {name}')
                    return self.simd_vec
                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def store_reduction(name, index, value):
                return self.store_reduction(name, index, value)

            @staticmethod
            def constant(val, dtype):
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    i32_iinfo = torch.iinfo(torch.int32)
                    if dtype == torch.int64 and val <= i32_iinfo.max and (val >= i32_iinfo.min):
                        opt_ctx.dtype = torch.int32
                    f32_iinfo = torch.finfo(torch.float32)
                    if dtype == torch.double:
                        if val <= f32_iinfo.max and val >= f32_iinfo.min or val == torch.inf or val == -torch.inf:
                            opt_ctx.dtype = torch.float32
                    supported_dtypes = [torch.float32, torch.int32, torch.bfloat16, torch.float16]
                    if opt_ctx.dtype not in supported_dtypes or (opt_ctx.dtype == torch.int32 and (not all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)))):
                        self.disable_vec(f'constant dtype: {opt_ctx.dtype}')
                    return val

            @staticmethod
            def index_expr(expr, dtype):
                assert len(self.ranges) == len(self.itervars)
                if not len(self.ranges) or not all((not isinstance(range, sympy.Expr) or sympy.simplify(range).is_number for range in self.ranges)):
                    self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
                    return self.cse.newvar()

                def can_use_int32():
                    free_symbols = list(expr.free_symbols)
                    sizes = {k: v for k, v in zip(self.itervars, self.ranges) if k in free_symbols}
                    if any((v == 0 for v in sizes.values())):
                        return True
                    vars_ranges = {k: ValueRanges(0, v - 1) for k, v in sizes.items()}
                    if not vars_ranges or len(vars_ranges) != len(free_symbols):
                        i32_iinfo = torch.iinfo(torch.int32)
                        return expr.is_number and expr <= i32_iinfo.max and (expr >= i32_iinfo.min)
                    expr_ranges = bound_sympy(expr, vars_ranges)
                    if math.isinf(expr_ranges.lower) or math.isinf(expr_ranges.upper):
                        return False
                    return range_expressable_in_32_bits(ValueRanges(int(expr_ranges.lower), int(expr_ranges.upper) + 1))
                with RecordOptimizationContext(__name__) as node_ctx:
                    assert len(self.ranges) == len(self.itervars)
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    if dtype == torch.int64 and can_use_int32() and all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)):
                        opt_ctx.dtype = torch.int32
                    else:
                        opt_ctx.dtype = dtype
                        self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
                    tiling_var = self.itervars[self.tiling_idx]
                    tiling_var_irrelevant = not expr.has(tiling_var)
                    if not tiling_var_irrelevant:
                        self.disable_vec(f'index_expr (tiling var relevant): {expr}, dtype {dtype}')
                    opt_ctx.is_most_inner_loop_irrevelant = tiling_var_irrelevant
                    tmp_var = self.cse.newvar()
                    return tmp_var

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                return sympy_symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                body()
                return self.cse.newvar()

            @staticmethod
            def to_dtype(x, dtype, src_dtype=None):
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    opt_ctx.dtype = dtype
                    cur_node = node_ctx.get_fx_node()
                    input_value: torch.fx.Node = cur_node.all_input_nodes[1]
                    if dtype == torch.float:
                        if input_value.target in ['load']:
                            dtype = V.graph.get_dtype(input_value.args[1]) if input_value.target == 'load' else input_value.args[-1]
                            if dtype in [torch.float16, torch.bfloat16, torch.float, torch.uint8]:
                                pass
                            elif dtype in [torch.int32, torch.int64] and input_value.target == 'load':
                                buffer = V.graph.get_buffer(input_value.args[1])
                                if not (isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0)):
                                    self.disable_vec(f'to_dtype: dtype {dtype}')
                            else:
                                self.disable_vec(f'to_dtype: dtype {dtype}')
                    elif dtype in DTYPE_LOWP_FP:
                        if not all((usr.target == 'store' for usr in cur_node.users)):
                            self.disable_vec('to_dtype: bfloat16/float16 expecting users are all stores')
                            return x
                        store_names = [usr.args[1] for usr in cur_node.users]
                        if not all((V.graph.get_dtype(name) in [dtype] for name in store_names)):
                            self.disable_vec('to_dtype: expecting all stores into bfloat16 or float16')
                            return x
                    elif dtype == torch.bool:
                        pass
                    elif dtype == torch.uint8:
                        is_to_uint8_and_store = all((usr.target in ['store'] for usr in cur_node.users))
                        is_to_uint8_and_to_float = all((usr.target in ['to_dtype'] and usr.args[2] == torch.float32 for usr in cur_node.users))
                        if not (is_to_uint8_and_store or is_to_uint8_and_to_float):
                            self.disable_vec(f'to_dtype: dtype {dtype}')
                    else:
                        self.disable_vec(f'to_dtype: dtype {dtype}')
                    return x
        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self