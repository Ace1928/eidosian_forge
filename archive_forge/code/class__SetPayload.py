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
class _SetPayload(object):

    def __init__(self, context, builder, set_type, ptr):
        payload = get_payload_struct(context, builder, set_type, ptr)
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._payload = payload
        self._entries = payload._get_ptr_by_name('entries')
        self._ptr = ptr

    @property
    def mask(self):
        return self._payload.mask

    @mask.setter
    def mask(self, value):
        self._payload.mask = value

    @property
    def used(self):
        return self._payload.used

    @used.setter
    def used(self, value):
        self._payload.used = value

    @property
    def fill(self):
        return self._payload.fill

    @fill.setter
    def fill(self, value):
        self._payload.fill = value

    @property
    def finger(self):
        return self._payload.finger

    @finger.setter
    def finger(self, value):
        self._payload.finger = value

    @property
    def dirty(self):
        return self._payload.dirty

    @dirty.setter
    def dirty(self, value):
        self._payload.dirty = value

    @property
    def entries(self):
        """
        A pointer to the start of the entries array.
        """
        return self._entries

    @property
    def ptr(self):
        """
        A pointer to the start of the NRT-allocated area.
        """
        return self._ptr

    def get_entry(self, idx):
        """
        Get entry number *idx*.
        """
        entry_ptr = cgutils.gep(self._builder, self._entries, idx)
        entry = self._context.make_data_helper(self._builder, types.SetEntry(self._ty), ref=entry_ptr)
        return entry

    def _lookup(self, item, h, for_insert=False):
        """
        Lookup the *item* with the given hash values in the entries.

        Return a (found, entry index) tuple:
        - If found is true, <entry index> points to the entry containing
          the item.
        - If found is false, <entry index> points to the empty entry that
          the item can be written to (only if *for_insert* is true)
        """
        context = self._context
        builder = self._builder
        intp_t = h.type
        mask = self.mask
        dtype = self._ty.dtype
        tyctx = context.typing_context
        fnty = tyctx.resolve_value_type(operator.eq)
        sig = fnty.get_call_type(tyctx, (dtype, dtype), {})
        eqfn = context.get_function(fnty, sig)
        one = ir.Constant(intp_t, 1)
        five = ir.Constant(intp_t, 5)
        perturb = cgutils.alloca_once_value(builder, h)
        index = cgutils.alloca_once_value(builder, builder.and_(h, mask))
        if for_insert:
            free_index_sentinel = mask.type(-1)
            free_index = cgutils.alloca_once_value(builder, free_index_sentinel)
        bb_body = builder.append_basic_block('lookup.body')
        bb_found = builder.append_basic_block('lookup.found')
        bb_not_found = builder.append_basic_block('lookup.not_found')
        bb_end = builder.append_basic_block('lookup.end')

        def check_entry(i):
            """
            Check entry *i* against the value being searched for.
            """
            entry = self.get_entry(i)
            entry_hash = entry.hash
            with builder.if_then(builder.icmp_unsigned('==', h, entry_hash)):
                eq = eqfn(builder, (item, entry.key))
                with builder.if_then(eq):
                    builder.branch(bb_found)
            with builder.if_then(is_hash_empty(context, builder, entry_hash)):
                builder.branch(bb_not_found)
            if for_insert:
                with builder.if_then(is_hash_deleted(context, builder, entry_hash)):
                    j = builder.load(free_index)
                    j = builder.select(builder.icmp_unsigned('==', j, free_index_sentinel), i, j)
                    builder.store(j, free_index)
        with cgutils.for_range(builder, ir.Constant(intp_t, LINEAR_PROBES)):
            i = builder.load(index)
            check_entry(i)
            i = builder.add(i, one)
            i = builder.and_(i, mask)
            builder.store(i, index)
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            i = builder.load(index)
            check_entry(i)
            p = builder.load(perturb)
            p = builder.lshr(p, five)
            i = builder.add(one, builder.mul(i, five))
            i = builder.and_(mask, builder.add(i, p))
            builder.store(i, index)
            builder.store(p, perturb)
            builder.branch(bb_body)
        with builder.goto_block(bb_not_found):
            if for_insert:
                i = builder.load(index)
                j = builder.load(free_index)
                i = builder.select(builder.icmp_unsigned('==', j, free_index_sentinel), i, j)
                builder.store(i, index)
            builder.branch(bb_end)
        with builder.goto_block(bb_found):
            builder.branch(bb_end)
        builder.position_at_end(bb_end)
        found = builder.phi(ir.IntType(1), 'found')
        found.add_incoming(cgutils.true_bit, bb_found)
        found.add_incoming(cgutils.false_bit, bb_not_found)
        return (found, builder.load(index))

    @contextlib.contextmanager
    def _iterate(self, start=None):
        """
        Iterate over the payload's entries.  Yield a SetLoop.
        """
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        one = ir.Constant(intp_t, 1)
        size = builder.add(self.mask, one)
        with cgutils.for_range(builder, size, start=start) as range_loop:
            entry = self.get_entry(range_loop.index)
            is_used = is_hash_used(context, builder, entry.hash)
            with builder.if_then(is_used):
                loop = SetLoop(index=range_loop.index, entry=entry, do_break=range_loop.do_break)
                yield loop

    @contextlib.contextmanager
    def _next_entry(self):
        """
        Yield a random entry from the payload.  Caller must ensure the
        set isn't empty, otherwise the function won't end.
        """
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        zero = ir.Constant(intp_t, 0)
        one = ir.Constant(intp_t, 1)
        mask = self.mask
        bb_body = builder.append_basic_block('next_entry_body')
        bb_end = builder.append_basic_block('next_entry_end')
        index = cgutils.alloca_once_value(builder, self.finger)
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            i = builder.load(index)
            i = builder.and_(mask, builder.add(i, one))
            builder.store(i, index)
            entry = self.get_entry(i)
            is_used = is_hash_used(context, builder, entry.hash)
            builder.cbranch(is_used, bb_end, bb_body)
        builder.position_at_end(bb_end)
        i = builder.load(index)
        self.finger = i
        yield self.get_entry(i)