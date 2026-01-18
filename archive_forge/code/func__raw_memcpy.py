import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def _raw_memcpy(builder, func_name, dst, src, count, itemsize, align):
    size_t = count.type
    if isinstance(itemsize, int):
        itemsize = ir.Constant(size_t, itemsize)
    memcpy = builder.module.declare_intrinsic(func_name, [voidptr_t, voidptr_t, size_t])
    is_volatile = false_bit
    builder.call(memcpy, [builder.bitcast(dst, voidptr_t), builder.bitcast(src, voidptr_t), builder.mul(count, itemsize), is_volatile])