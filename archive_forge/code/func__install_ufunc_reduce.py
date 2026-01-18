import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def _install_ufunc_reduce(self, template) -> None:
    at = types.Function(template)

    @overload_method(at, 'reduce')
    def ol_reduce(ufunc, array, axis=0, dtype=None, initial=None):
        warnings.warn('ufunc.reduce feature is experimental', category=errors.NumbaExperimentalFeatureWarning)
        if not isinstance(array, types.Array):
            msg = 'The first argument "array" must be array-like'
            raise errors.NumbaTypeError(msg)
        axis_int = isinstance(axis, types.Integer)
        axis_int_tuple = isinstance(axis, types.UniTuple) and isinstance(axis.dtype, types.Integer)
        axis_empty_tuple = isinstance(axis, types.Tuple) and len(axis) == 0
        axis_none = cgutils.is_nonelike(axis)
        axis_tuple_size = len(axis) if axis_int_tuple else 0
        if self.ufunc.identity is None and (not (axis_int_tuple and axis_tuple_size == 1 or axis_empty_tuple or axis_int or axis_none)):
            msg = f"reduction operation '{self.ufunc.__name__}' is not reorderable, so at most one axis may be specified"
            raise errors.NumbaTypeError(msg)
        tup_init = (0,) * array.ndim
        tup_init_m1 = (0,) * (array.ndim - 1)
        nb_dtype = array.dtype if cgutils.is_nonelike(dtype) else dtype
        identity = self.identity
        id_none = cgutils.is_nonelike(identity)
        init_none = cgutils.is_nonelike(initial)

        @register_jitable
        def tuple_slice(tup, pos):
            s = tup_init_m1
            i = 0
            for j, e in enumerate(tup):
                if j == pos:
                    continue
                s = tuple_setitem(s, i, e)
                i += 1
            return s

        @register_jitable
        def tuple_slice_append(tup, pos, val):
            s = tup_init
            i, j, sz = (0, 0, len(s))
            while j < sz:
                if j == pos:
                    s = tuple_setitem(s, j, val)
                else:
                    e = tup[i]
                    s = tuple_setitem(s, j, e)
                    i += 1
                j += 1
            return s

        @intrinsic
        def compute_flat_idx(typingctx, strides, itemsize, idx, axis):
            sig = types.intp(strides, itemsize, idx, axis)
            len_idx = len(idx)

            def gen_block(builder, block_pos, block_name, bb_end, args):
                strides, _, idx, _ = args
                bb = builder.append_basic_block(name=block_name)
                with builder.goto_block(bb):
                    zero = ir.IntType(64)(0)
                    flat_idx = zero
                    if block_pos == 0:
                        for i in range(1, len_idx):
                            stride = builder.extract_value(strides, i - 1)
                            idx_i = builder.extract_value(idx, i)
                            m = builder.mul(stride, idx_i)
                            flat_idx = builder.add(flat_idx, m)
                    elif 0 < block_pos < len_idx - 1:
                        for i in range(0, block_pos):
                            stride = builder.extract_value(strides, i)
                            idx_i = builder.extract_value(idx, i)
                            m = builder.mul(stride, idx_i)
                            flat_idx = builder.add(flat_idx, m)
                        for i in range(block_pos + 1, len_idx):
                            stride = builder.extract_value(strides, i - 1)
                            idx_i = builder.extract_value(idx, i)
                            m = builder.mul(stride, idx_i)
                            flat_idx = builder.add(flat_idx, m)
                    else:
                        for i in range(0, len_idx - 1):
                            stride = builder.extract_value(strides, i)
                            idx_i = builder.extract_value(idx, i)
                            m = builder.mul(stride, idx_i)
                            flat_idx = builder.add(flat_idx, m)
                    builder.branch(bb_end)
                return (bb, flat_idx)

            def codegen(context, builder, sig, args):
                strides, itemsize, idx, axis = args
                bb = builder.basic_block
                switch_end = builder.append_basic_block(name='axis_end')
                l = []
                for i in range(len_idx):
                    block, flat_idx = gen_block(builder, i, f'axis_{i}', switch_end, args)
                    l.append((block, flat_idx))
                with builder.goto_block(bb):
                    switch = builder.switch(axis, l[-1][0])
                    for i in range(len_idx):
                        switch.add_case(i, l[i][0])
                builder.position_at_end(switch_end)
                phi = builder.phi(l[0][1].type)
                for block, value in l:
                    phi.add_incoming(value, block)
                return builder.sdiv(phi, itemsize)
            return (sig, codegen)

        @register_jitable
        def fixup_axis(axis, ndim):
            ax = axis
            for i in range(len(axis)):
                val = axis[i] + ndim if axis[i] < 0 else axis[i]
                ax = tuple_setitem(ax, i, val)
            return ax

        @register_jitable
        def find_min(tup):
            idx, e = (0, tup[0])
            for i in range(len(tup)):
                if tup[i] < e:
                    idx, e = (i, tup[i])
            return (idx, e)

        def impl_1d(ufunc, array, axis=0, dtype=None, initial=None):
            start = 0
            if init_none and id_none:
                start = 1
                r = array[0]
            elif init_none:
                r = identity
            else:
                r = initial
            sz = array.shape[0]
            for i in range(start, sz):
                r = ufunc(r, array[i])
            return r

        def impl_nd_axis_int(ufunc, array, axis=0, dtype=None, initial=None):
            if axis is None:
                raise ValueError("'axis' must be specified")
            if axis < 0:
                axis += array.ndim
            if axis < 0 or axis >= array.ndim:
                raise ValueError('Invalid axis')
            shape = tuple_slice(array.shape, axis)
            if initial is None and identity is None:
                r = np.empty(shape, dtype=nb_dtype)
                for idx, _ in np.ndenumerate(r):
                    result_idx = tuple_slice_append(idx, axis, 0)
                    r[idx] = array[result_idx]
            elif initial is None and identity is not None:
                r = np.full(shape, fill_value=identity, dtype=nb_dtype)
            else:
                r = np.full(shape, fill_value=initial, dtype=nb_dtype)
            view = r.ravel()
            if initial is None and identity is None:
                for idx, val in np.ndenumerate(array):
                    if idx[axis] == 0:
                        continue
                    else:
                        flat_pos = compute_flat_idx(r.strides, r.itemsize, idx, axis)
                        lhs, rhs = (view[flat_pos], val)
                        view[flat_pos] = ufunc(lhs, rhs)
            else:
                for idx, val in np.ndenumerate(array):
                    if initial is None and identity is None and (idx[axis] == 0):
                        continue
                    flat_pos = compute_flat_idx(r.strides, r.itemsize, idx, axis)
                    lhs, rhs = (view[flat_pos], val)
                    view[flat_pos] = ufunc(lhs, rhs)
            return r

        def impl_nd_axis_tuple(ufunc, array, axis=0, dtype=None, initial=None):
            axis_ = fixup_axis(axis, array.ndim)
            for i in range(0, len(axis_)):
                if axis_[i] < 0 or axis_[i] >= array.ndim:
                    raise ValueError('Invalid axis')
                for j in range(i + 1, len(axis_)):
                    if axis_[i] == axis_[j]:
                        raise ValueError("duplicate value in 'axis'")
            min_idx, min_elem = find_min(axis_)
            r = ufunc.reduce(array, axis=min_elem, dtype=dtype, initial=initial)
            if len(axis) == 1:
                return r
            elif len(axis) == 2:
                return ufunc.reduce(r, axis=axis_[(min_idx + 1) % 2] - 1)
            else:
                ax = axis_tup
                for i in range(len(ax)):
                    if i != min_idx:
                        ax = tuple_setitem(ax, i, axis_[i])
                return ufunc.reduce(r, axis=ax)

        def impl_axis_empty_tuple(ufunc, array, axis=0, dtype=None, initial=None):
            return array

        def impl_axis_none(ufunc, array, axis=0, dtype=None, initial=None):
            return ufunc.reduce(array, axis_tup, dtype, initial)
        if array.ndim == 1 and (not axis_empty_tuple):
            return impl_1d
        elif axis_empty_tuple:
            return impl_axis_empty_tuple
        elif axis_none:
            axis_tup = tuple(range(array.ndim))
            return impl_axis_none
        elif axis_int_tuple:
            axis_tup = (0,) * (len(axis) - 1)
            return impl_nd_axis_tuple
        elif axis == 0 or isinstance(axis, (types.Integer, types.Omitted, types.IntegerLiteral)):
            return impl_nd_axis_int