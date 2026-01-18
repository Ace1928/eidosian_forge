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
class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads, tiling_factor=0, tiling_idx=-1, tiling_dtype=torch.float):
        super().__init__(args, num_threads)
        assert codecache.pick_vec_isa()
        if tiling_factor == 0:
            tiling_factor = codecache.pick_vec_isa().nelements(dtype=tiling_dtype)
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx
        metrics.generated_cpp_vec_kernel_count += 1

    def load(self, name: str, index: sympy.Expr):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)
        dtype = V.graph.get_dtype(name)
        tiling_var = self.itervars[self.tiling_idx]
        is_broadcast = not index.has(tiling_var)
        is_mask = dtype in [torch.bool, torch.uint8] and (not opt_ctx.is_load_uint8_as_float)
        load_mask = f'to_float_mask({self._load_mask})' if self._load_mask else None
        non_contiguous = not is_broadcast and stride_at(tiling_var, index) != 1 or any((self.cse.varname_map[s.name].depends_on(tiling_var) for s in index.free_symbols if s.name.startswith('tmp')))
        var_expr = f'{var}[{cexpr_index(index)}]' if is_broadcast else f'{var} + {cexpr_index(index)}'
        loadbuf = 'tmpbuf' if non_contiguous else var_expr
        if is_broadcast:
            csevar = super().load(name, index)
            csevar.dtype = dtype
            return csevar
        elif dtype in [torch.uint8] and opt_ctx.is_load_uint8_as_float:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<uint8_t>::loadu_one_fourth({loadbuf})'
        elif is_mask:
            line = f'flag_to_float_vec({loadbuf})'
        elif dtype in DTYPE_LOWP_FP:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>::loadu({loadbuf}, {self.tiling_factor})'
        else:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<float>::loadu({loadbuf})'
        if non_contiguous:
            tmpbuftype = 'float' if is_mask else f'{DTYPE_TO_CPP[dtype]}'
            tmpbufsize = f'{self.tiling_factor}'
            if dtype in DTYPE_LOWP_FP:
                tmpbufsize += ' * 2'
            tmpbufdeclare = f'__at_align__ {tmpbuftype} tmpbuf[{tmpbufsize}];'
            inner = sympy_symbol(f'{tiling_var}_inner')
            new_index = self.scale_index_with_offset(index, itervar_idx=self.tiling_idx, offset=inner)
            tmpbufdefine = f'for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) '
            rhs = f'{var}[{cexpr_index(new_index)}]'
            if is_mask:
                rhs = f'flag_to_float_scalar({rhs})'
            tmpbufdefine += f'tmpbuf[{inner}] = {rhs};'
            line = f'([&]() {{ {tmpbufdeclare} {tmpbufdefine} return {line}; }})()'
        csevar = self.cse.generate(self.loads, line)
        csevar.update_on_args('load', (name, index), {})
        assert isinstance(csevar, CppCSEVariable)
        csevar.is_vec = True
        return csevar

    def get_vec_store_line(self, value, var, index, dtype):
        """
        Get a store line str that stores `value` into `var` at `index` of `dtype`.
        :param value: Vectorized type templaterized on `dtype`.
        :param var: buffer to store into.
        :index: index into the `var`.
        """
        assert isinstance(value, str) or (isinstance(value, CppCSEVariable) and value.is_vec), value
        tiling_var = self.itervars[self.tiling_idx]
        assert index.has(tiling_var)
        var_expr = f'{var} + {cexpr_index(index)}'
        non_contiguous = stride_at(tiling_var, index) != 1 or 'tmp' in f'{index}'
        if non_contiguous:
            var_expr = 'tmpbuf'
        if dtype == torch.float:
            line = f'{value}.store({var_expr});'
        else:
            line = f'{value}.store({var_expr}, {self.tiling_factor});'
        if non_contiguous:
            inner = sympy_symbol(f'{tiling_var}_inner')
            new_index = self.scale_index_with_offset(index, itervar_idx=self.tiling_idx, offset=inner)
            tmp_bufsize = f'{self.tiling_factor}*sizeof(float)/sizeof({DTYPE_TO_CPP[dtype]})'
            line = f'{{ __at_align__ {DTYPE_TO_CPP[dtype]} tmpbuf[{tmp_bufsize}]; {line} for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) {var}[{cexpr_index(new_index)}] = tmpbuf[{inner}]; }}'
        return line

    def store(self, name, index, value, mode=None):
        assert 'buf' in name
        assert mode is None
        assert isinstance(value, CppCSEVariable), value
        if not value.is_vec:
            value = self.broadcast(value)
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)
        index = self.rename_indexing(index)
        self.stores.writeline(DeferredLine(name, self.get_vec_store_line(value, var, index, V.graph.get_dtype(name))))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        assert reduction_type in {'max', 'min', 'sum', 'prod', 'xor_sum', 'welford_reduce', 'welford_combine'}
        assert dtype == torch.float
        assert src_dtype == torch.float
        assert isinstance(value, CppCSEVariable) and value.is_vec, value
        vec_ns = 'at::vec'
        vec = f'{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>'
        acc_type = reduction_acc_type(reduction_type, dtype)
        acc_type_vec = reduction_acc_type_vec(reduction_type, dtype)
        if (reduction_type, acc_type) not in self.reduction_omp_dec:
            if RTYPE_TO_CPP[reduction_type] not in NATIVE_OMP_RTYPES:
                self.reduction_prefix.splice(f'#pragma omp declare reduction({RTYPE_TO_CPP[reduction_type]}:{acc_type}:omp_out = {reduction_combine(reduction_type, 'omp_out', 'omp_in')}) initializer(omp_priv={{{reduction_init(reduction_type, dtype)}}})\n            ')
            self.reduction_omp_dec[reduction_type, acc_type] = RTYPE_TO_CPP[reduction_type]
        if (reduction_type, acc_type_vec) not in self.reduction_omp_dec:
            self.reduction_prefix.splice(f'#pragma omp declare reduction({RTYPE_TO_CPP[reduction_type]}:{acc_type_vec}:omp_out = {reduction_combine_vec(reduction_type, 'omp_out', 'omp_in')}) initializer(omp_priv={{{reduction_init_vec(reduction_type, dtype)}}})\n            ')
            self.reduction_omp_dec[reduction_type, acc_type_vec] = RTYPE_TO_CPP[reduction_type]
        reduction_key = (src_dtype, reduction_type, value)
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]
        acc = self.reduction_cse.generate(self.loads, f'reduction {reduction_key}', write=False)
        acc_vec = f'{acc}_vec'
        self.reduction_var_map[acc_vec] = reduction_type
        self.reduction_prefix.writeline(f'{acc_type} {acc} = {reduction_init(reduction_type, dtype)};')
        self.reduction_prefix.writeline(f'{acc_type_vec} {acc_vec} = {reduction_init_vec(reduction_type, dtype)};')
        self.stores.writeline(f'{acc_vec} = {reduction_combine_vec(reduction_type, acc_vec, value)};')
        tmpvar: Union[str, CSEVariable]
        if self.tiling_idx >= self.reduction_depth:
            if is_welford_reduction(reduction_type):
                next_value = f'welford_vec_reduce_all({acc_vec})'
            else:
                reduce_all_body = '{ return ' + reduction_combine_vec(reduction_type, 'x', 'y') + '; }'
                vec_reduce_all_func = f'{vec_ns}::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>'
                next_value = f'{vec_reduce_all_func}([]({vec}& x, {vec}& y) {reduce_all_body}, {acc_vec})'
            self.reduction_suffix.writeline(f'{acc} = {reduction_combine(reduction_type, acc, next_value)};')
            tmpvar = acc
        else:
            tmpvar = acc_vec
        result = reduction_project(reduction_type, tmpvar)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        out_dtype = V.graph.get_dtype(name)
        dtype = torch.float
        if self.tiling_idx >= self.reduction_depth:
            self.reduction_suffix.writeline(DeferredLine(name, f'{var}[{cexpr_index(index)}] = static_cast<{DTYPE_TO_CPP[out_dtype]}>({value});'))
        else:
            store_lines = []
            if out_dtype != dtype:
                if out_dtype in DTYPE_LOWP_FP and dtype == torch.float:
                    _lowp_fp_tmpvar_vec = f'{DTYPE_TO_CPP[out_dtype]}_{value}'
                    store_lines = [DeferredLine(name, f'auto {_lowp_fp_tmpvar_vec} = cvt_fp32_to_lowp_fp<{DTYPE_TO_CPP[out_dtype]}>({value});')]
                    value = _lowp_fp_tmpvar_vec
                else:
                    raise AssertionError(f'Unsupported reduction type from {dtype} to {out_dtype}')
            store_lines += [DeferredLine(name, self.get_vec_store_line(value, var, index, out_dtype))]
            self.reduction_suffix.writelines(store_lines)

    def broadcast(self, scalar_var: CppCSEVariable):
        assert not scalar_var.is_vec and self.itervars[self.tiling_idx] not in scalar_var.dependent_itervars
        if scalar_var.dtype == torch.bool:
            vec_var = self.cse.generate(self.compute, f'to_float_mask({scalar_var.name})')
        else:
            assert scalar_var.dtype is not None
            vec_var = self.cse.generate(self.compute, f'at::vec::Vectorized<{DTYPE_TO_CPP[scalar_var.dtype]}>({scalar_var.name})')
        assert isinstance(vec_var, CppCSEVariable)
        vec_var.dtype = scalar_var.dtype
        vec_var.dependent_itervars = scalar_var.dependent_itervars
        vec_var.is_vec = True
        return vec_var