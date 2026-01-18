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
class SetInstance(object):

    def __init__(self, context, builder, set_type, set_val):
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._entrysize = get_entry_size(context, set_type)
        self._set = context.make_helper(builder, set_type, set_val)

    @property
    def dtype(self):
        return self._ty.dtype

    @property
    def payload(self):
        """
        The _SetPayload for this set.
        """
        context = self._context
        builder = self._builder
        ptr = self._context.nrt.meminfo_data(builder, self.meminfo)
        return _SetPayload(context, builder, self._ty, ptr)

    @property
    def value(self):
        return self._set._getvalue()

    @property
    def meminfo(self):
        return self._set.meminfo

    @property
    def parent(self):
        return self._set.parent

    @parent.setter
    def parent(self, value):
        self._set.parent = value

    def get_size(self):
        """
        Return the number of elements in the size.
        """
        return self.payload.used

    def set_dirty(self, val):
        if self._ty.reflected:
            self.payload.dirty = cgutils.true_bit if val else cgutils.false_bit

    def _add_entry(self, payload, entry, item, h, do_resize=True):
        context = self._context
        builder = self._builder
        old_hash = entry.hash
        entry.hash = h
        self.incref_value(item)
        entry.key = item
        used = payload.used
        one = ir.Constant(used.type, 1)
        used = payload.used = builder.add(used, one)
        with builder.if_then(is_hash_empty(context, builder, old_hash), likely=True):
            payload.fill = builder.add(payload.fill, one)
        if do_resize:
            self.upsize(used)
        self.set_dirty(True)

    def _add_key(self, payload, item, h, do_resize=True, do_incref=True):
        context = self._context
        builder = self._builder
        found, i = payload._lookup(item, h, for_insert=True)
        not_found = builder.not_(found)
        with builder.if_then(not_found):
            entry = payload.get_entry(i)
            old_hash = entry.hash
            entry.hash = h
            if do_incref:
                self.incref_value(item)
            entry.key = item
            used = payload.used
            one = ir.Constant(used.type, 1)
            used = payload.used = builder.add(used, one)
            with builder.if_then(is_hash_empty(context, builder, old_hash), likely=True):
                payload.fill = builder.add(payload.fill, one)
            if do_resize:
                self.upsize(used)
            self.set_dirty(True)

    def _remove_entry(self, payload, entry, do_resize=True, do_decref=True):
        entry.hash = ir.Constant(entry.hash.type, DELETED)
        if do_decref:
            self.decref_value(entry.key)
        used = payload.used
        one = ir.Constant(used.type, 1)
        used = payload.used = self._builder.sub(used, one)
        if do_resize:
            self.downsize(used)
        self.set_dirty(True)

    def _remove_key(self, payload, item, h, do_resize=True):
        context = self._context
        builder = self._builder
        found, i = payload._lookup(item, h)
        with builder.if_then(found):
            entry = payload.get_entry(i)
            self._remove_entry(payload, entry, do_resize)
        return found

    def add(self, item, do_resize=True):
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        self._add_key(payload, item, h, do_resize)

    def add_pyapi(self, pyapi, item, do_resize=True):
        """A version of .add for use inside functions following Python calling
        convention.
        """
        context = self._context
        builder = self._builder
        payload = self.payload
        h = self._pyapi_get_hash_value(pyapi, context, builder, item)
        self._add_key(payload, item, h, do_resize)

    def _pyapi_get_hash_value(self, pyapi, context, builder, item):
        """Python API compatible version of `get_hash_value()`.
        """
        argtypes = [self._ty.dtype]
        resty = types.intp

        def wrapper(val):
            return _get_hash_value_intrinsic(val)
        args = [item]
        sig = typing.signature(resty, *argtypes)
        is_error, retval = pyapi.call_jit_code(wrapper, sig, args)
        with builder.if_then(is_error, likely=False):
            builder.ret(pyapi.get_null_object())
        return retval

    def contains(self, item):
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        found, i = payload._lookup(item, h)
        return found

    def discard(self, item):
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        found = self._remove_key(payload, item, h)
        return found

    def pop(self):
        context = self._context
        builder = self._builder
        lty = context.get_value_type(self._ty.dtype)
        key = cgutils.alloca_once(builder, lty)
        payload = self.payload
        with payload._next_entry() as entry:
            builder.store(entry.key, key)
            self._remove_entry(payload, entry, do_decref=False)
        return builder.load(key)

    def clear(self):
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        minsize = ir.Constant(intp_t, MINSIZE)
        self._replace_payload(minsize)
        self.set_dirty(True)

    def copy(self):
        """
        Return a copy of this set.
        """
        context = self._context
        builder = self._builder
        payload = self.payload
        used = payload.used
        fill = payload.fill
        other = type(self)(context, builder, self._ty, None)
        no_deleted_entries = builder.icmp_unsigned('==', used, fill)
        with builder.if_else(no_deleted_entries, likely=True) as (if_no_deleted, if_deleted):
            with if_no_deleted:
                ok = other._copy_payload(payload)
                with builder.if_then(builder.not_(ok), likely=False):
                    context.call_conv.return_user_exc(builder, MemoryError, ('cannot copy set',))
            with if_deleted:
                nentries = self.choose_alloc_size(context, builder, used)
                ok = other._allocate_payload(nentries)
                with builder.if_then(builder.not_(ok), likely=False):
                    context.call_conv.return_user_exc(builder, MemoryError, ('cannot copy set',))
                other_payload = other.payload
                with payload._iterate() as loop:
                    entry = loop.entry
                    other._add_key(other_payload, entry.key, entry.hash, do_resize=False)
        return other

    def intersect(self, other):
        """
        In-place intersection with *other* set.
        """
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        with payload._iterate() as loop:
            entry = loop.entry
            found, _ = other_payload._lookup(entry.key, entry.hash)
            with builder.if_then(builder.not_(found)):
                self._remove_entry(payload, entry, do_resize=False)
        self.downsize(payload.used)

    def difference(self, other):
        """
        In-place difference with *other* set.
        """
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        with other_payload._iterate() as loop:
            entry = loop.entry
            self._remove_key(payload, entry.key, entry.hash, do_resize=False)
        self.downsize(payload.used)

    def symmetric_difference(self, other):
        """
        In-place symmetric difference with *other* set.
        """
        context = self._context
        builder = self._builder
        other_payload = other.payload
        with other_payload._iterate() as loop:
            key = loop.entry.key
            h = loop.entry.hash
            payload = self.payload
            found, i = payload._lookup(key, h, for_insert=True)
            entry = payload.get_entry(i)
            with builder.if_else(found) as (if_common, if_not_common):
                with if_common:
                    self._remove_entry(payload, entry, do_resize=False)
                with if_not_common:
                    self._add_entry(payload, entry, key, h)
        self.downsize(self.payload.used)

    def issubset(self, other, strict=False):
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        cmp_op = '<' if strict else '<='
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)
        with builder.if_else(builder.icmp_unsigned(cmp_op, payload.used, other_payload.used)) as (if_smaller, if_larger):
            with if_larger:
                builder.store(cgutils.false_bit, res)
            with if_smaller:
                with payload._iterate() as loop:
                    entry = loop.entry
                    found, _ = other_payload._lookup(entry.key, entry.hash)
                    with builder.if_then(builder.not_(found)):
                        builder.store(cgutils.false_bit, res)
                        loop.do_break()
        return builder.load(res)

    def isdisjoint(self, other):
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)

        def check(smaller, larger):
            with smaller._iterate() as loop:
                entry = loop.entry
                found, _ = larger._lookup(entry.key, entry.hash)
                with builder.if_then(found):
                    builder.store(cgutils.false_bit, res)
                    loop.do_break()
        with builder.if_else(builder.icmp_unsigned('>', payload.used, other_payload.used)) as (if_larger, otherwise):
            with if_larger:
                check(other_payload, payload)
            with otherwise:
                check(payload, other_payload)
        return builder.load(res)

    def equals(self, other):
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)
        with builder.if_else(builder.icmp_unsigned('==', payload.used, other_payload.used)) as (if_same_size, otherwise):
            with if_same_size:
                with payload._iterate() as loop:
                    entry = loop.entry
                    found, _ = other_payload._lookup(entry.key, entry.hash)
                    with builder.if_then(builder.not_(found)):
                        builder.store(cgutils.false_bit, res)
                        loop.do_break()
            with otherwise:
                builder.store(cgutils.false_bit, res)
        return builder.load(res)

    @classmethod
    def allocate_ex(cls, context, builder, set_type, nitems=None):
        """
        Allocate a SetInstance with its storage.
        Return a (ok, instance) tuple where *ok* is a LLVM boolean and
        *instance* is a SetInstance object (the object's contents are
        only valid when *ok* is true).
        """
        intp_t = context.get_value_type(types.intp)
        if nitems is None:
            nentries = ir.Constant(intp_t, MINSIZE)
        else:
            if isinstance(nitems, int):
                nitems = ir.Constant(intp_t, nitems)
            nentries = cls.choose_alloc_size(context, builder, nitems)
        self = cls(context, builder, set_type, None)
        ok = self._allocate_payload(nentries)
        return (ok, self)

    @classmethod
    def allocate(cls, context, builder, set_type, nitems=None):
        """
        Allocate a SetInstance with its storage.  Same as allocate_ex(),
        but return an initialized *instance*.  If allocation failed,
        control is transferred to the caller using the target's current
        call convention.
        """
        ok, self = cls.allocate_ex(context, builder, set_type, nitems)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot allocate set',))
        return self

    @classmethod
    def from_meminfo(cls, context, builder, set_type, meminfo):
        """
        Allocate a new set instance pointing to an existing payload
        (a meminfo pointer).
        Note the parent field has to be filled by the caller.
        """
        self = cls(context, builder, set_type, None)
        self._set.meminfo = meminfo
        self._set.parent = context.get_constant_null(types.pyobject)
        context.nrt.incref(builder, set_type, self.value)
        return self

    @classmethod
    def choose_alloc_size(cls, context, builder, nitems):
        """
        Choose a suitable number of entries for the given number of items.
        """
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        minsize = ir.Constant(intp_t, MINSIZE)
        min_entries = builder.shl(nitems, one)
        size_p = cgutils.alloca_once_value(builder, minsize)
        bb_body = builder.append_basic_block('calcsize.body')
        bb_end = builder.append_basic_block('calcsize.end')
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            size = builder.load(size_p)
            is_large_enough = builder.icmp_unsigned('>=', size, min_entries)
            with builder.if_then(is_large_enough, likely=False):
                builder.branch(bb_end)
            next_size = builder.shl(size, one)
            builder.store(next_size, size_p)
            builder.branch(bb_body)
        builder.position_at_end(bb_end)
        return builder.load(size_p)

    def upsize(self, nitems):
        """
        When adding to the set, ensure it is properly sized for the given
        number of used entries.
        """
        context = self._context
        builder = self._builder
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        two = ir.Constant(intp_t, 2)
        payload = self.payload
        min_entries = builder.shl(nitems, one)
        size = builder.add(payload.mask, one)
        need_resize = builder.icmp_unsigned('>=', min_entries, size)
        with builder.if_then(need_resize, likely=False):
            new_size_p = cgutils.alloca_once_value(builder, size)
            bb_body = builder.append_basic_block('calcsize.body')
            bb_end = builder.append_basic_block('calcsize.end')
            builder.branch(bb_body)
            with builder.goto_block(bb_body):
                new_size = builder.load(new_size_p)
                new_size = builder.shl(new_size, two)
                builder.store(new_size, new_size_p)
                is_too_small = builder.icmp_unsigned('>=', min_entries, new_size)
                builder.cbranch(is_too_small, bb_body, bb_end)
            builder.position_at_end(bb_end)
            new_size = builder.load(new_size_p)
            if DEBUG_ALLOCS:
                context.printf(builder, 'upsize to %zd items: current size = %zd, min entries = %zd, new size = %zd\n', nitems, size, min_entries, new_size)
            self._resize(payload, new_size, 'cannot grow set')

    def downsize(self, nitems):
        """
        When removing from the set, ensure it is properly sized for the given
        number of used entries.
        """
        context = self._context
        builder = self._builder
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        two = ir.Constant(intp_t, 2)
        minsize = ir.Constant(intp_t, MINSIZE)
        payload = self.payload
        min_entries = builder.shl(nitems, one)
        min_entries = builder.select(builder.icmp_unsigned('>=', min_entries, minsize), min_entries, minsize)
        max_size = builder.shl(min_entries, two)
        size = builder.add(payload.mask, one)
        need_resize = builder.and_(builder.icmp_unsigned('<=', max_size, size), builder.icmp_unsigned('<', minsize, size))
        with builder.if_then(need_resize, likely=False):
            new_size_p = cgutils.alloca_once_value(builder, size)
            bb_body = builder.append_basic_block('calcsize.body')
            bb_end = builder.append_basic_block('calcsize.end')
            builder.branch(bb_body)
            with builder.goto_block(bb_body):
                new_size = builder.load(new_size_p)
                new_size = builder.lshr(new_size, one)
                is_too_small = builder.icmp_unsigned('>', min_entries, new_size)
                with builder.if_then(is_too_small):
                    builder.branch(bb_end)
                builder.store(new_size, new_size_p)
                builder.branch(bb_body)
            builder.position_at_end(bb_end)
            new_size = builder.load(new_size_p)
            if DEBUG_ALLOCS:
                context.printf(builder, 'downsize to %zd items: current size = %zd, min entries = %zd, new size = %zd\n', nitems, size, min_entries, new_size)
            self._resize(payload, new_size, 'cannot shrink set')

    def _resize(self, payload, nentries, errmsg):
        """
        Resize the payload to the given number of entries.

        CAUTION: *nentries* must be a power of 2!
        """
        context = self._context
        builder = self._builder
        old_payload = payload
        ok = self._allocate_payload(nentries, realloc=True)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, (errmsg,))
        payload = self.payload
        with old_payload._iterate() as loop:
            entry = loop.entry
            self._add_key(payload, entry.key, entry.hash, do_resize=False, do_incref=False)
        self._free_payload(old_payload.ptr)

    def _replace_payload(self, nentries):
        """
        Replace the payload with a new empty payload with the given number
        of entries.

        CAUTION: *nentries* must be a power of 2!
        """
        context = self._context
        builder = self._builder
        with self.payload._iterate() as loop:
            entry = loop.entry
            self.decref_value(entry.key)
        self._free_payload(self.payload.ptr)
        ok = self._allocate_payload(nentries, realloc=True)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot reallocate set',))

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

    def _free_payload(self, ptr):
        """
        Free an allocated old payload at *ptr*.
        """
        self._context.nrt.meminfo_varsize_free(self._builder, self.meminfo, ptr)

    def _copy_payload(self, src_payload):
        """
        Raw-copy the given payload into self.
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
        mask = src_payload.mask
        nentries = builder.add(one, mask)
        allocsize = builder.add(ir.Constant(intp_t, payload_size), builder.mul(ir.Constant(intp_t, entry_size), nentries))
        with builder.if_then(builder.load(ok), likely=True):
            dtor = self._imp_dtor(context, builder.module)
            meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t))
            alloc_ok = cgutils.is_null(builder, meminfo)
            with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    self._set.meminfo = meminfo
                    payload = self.payload
                    payload.used = src_payload.used
                    payload.fill = src_payload.fill
                    payload.finger = zero
                    payload.mask = mask
                    cgutils.raw_memcpy(builder, payload.entries, src_payload.entries, nentries, entry_size)
                    with src_payload._iterate() as loop:
                        self.incref_value(loop.entry.key)
                    if DEBUG_ALLOCS:
                        context.printf(builder, 'allocated %zd bytes for set at %p: mask = %zd\n', allocsize, payload.ptr, mask)
        return builder.load(ok)

    def _imp_dtor(self, context, module):
        """Define the dtor for set
        """
        llvoidptr = cgutils.voidptr_t
        llsize_t = context.get_value_type(types.size_t)
        fnty = ir.FunctionType(ir.VoidType(), [llvoidptr, llsize_t, llvoidptr])
        fname = f'.dtor.set.{self._ty.dtype}'
        fn = cgutils.get_or_insert_function(module, fnty, name=fname)
        if fn.is_declaration:
            fn.linkage = 'linkonce_odr'
            builder = ir.IRBuilder(fn.append_basic_block())
            payload = _SetPayload(context, builder, self._ty, fn.args[0])
            with payload._iterate() as loop:
                entry = loop.entry
                context.nrt.decref(builder, self._ty.dtype, entry.key)
            builder.ret_void()
        return fn

    def incref_value(self, val):
        """Incref an element value
        """
        self._context.nrt.incref(self._builder, self._ty.dtype, val)

    def decref_value(self, val):
        """Decref an element value
        """
        self._context.nrt.decref(self._builder, self._ty.dtype, val)