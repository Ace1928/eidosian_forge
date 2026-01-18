import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
def _allocate_payload(self, nentries, realloc=False):
    """
        Allocate and initialize payload for the given number of entries.
        If *realloc* is True, the existing meminfo is reused.

        CAUTION: *nentries* must be a power of 2!
        """
    context = self._context
    builder = self._builder
    ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
    intp_t = context.get_value_type(types.intp)
    zero = ir.Constant(intp_t, 0)
    one = ir.Constant(intp_t, 1)
    payload_type = context.get_data_type(types.SetPayload(self._ty))
    payload_size = context.get_abi_sizeof(payload_type)
    entry_size = self._entrysize
    payload_size -= entry_size
    allocsize, ovf = cgutils.muladd_with_overflow(builder, nentries, ir.Constant(intp_t, entry_size), ir.Constant(intp_t, payload_size))
    with builder.if_then(ovf, likely=False):
        builder.store(cgutils.false_bit, ok)
    with builder.if_then(builder.load(ok), likely=True):
        if realloc:
            meminfo = self._set.meminfo
            ptr = context.nrt.meminfo_varsize_alloc_unchecked(builder, meminfo, size=allocsize)
            alloc_ok = cgutils.is_null(builder, ptr)
        else:
            dtor = self._imp_dtor(context, builder.module)
            meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t))
            alloc_ok = cgutils.is_null(builder, meminfo)
        with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
            with if_error:
                builder.store(cgutils.false_bit, ok)
            with if_ok:
                if not realloc:
                    self._set.meminfo = meminfo
                    self._set.parent = context.get_constant_null(types.pyobject)
                payload = self.payload
                cgutils.memset(builder, payload.ptr, allocsize, 255)
                payload.used = zero
                payload.fill = zero
                payload.finger = zero
                new_mask = builder.sub(nentries, one)
                payload.mask = new_mask
                if DEBUG_ALLOCS:
                    context.printf(builder, 'allocated %zd bytes for set at %p: mask = %zd\n', allocsize, payload.ptr, new_mask)
    return builder.load(ok)