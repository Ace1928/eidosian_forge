from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
@unbox(types.ClassInstanceType)
def _unbox_class_instance(typ, val, c):

    def access_member(member_offset):
        offset = c.context.get_constant(types.uintp, member_offset)
        llvoidptr = ir.IntType(8).as_pointer()
        ptr = cgutils.pointer_add(c.builder, val, offset)
        casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
        return c.builder.load(casted)
    struct_cls = cgutils.create_struct_proxy(typ)
    inst = struct_cls(c.context, c.builder)
    ptr_meminfo = access_member(_box.box_meminfoptr_offset)
    ptr_dataptr = access_member(_box.box_dataptr_offset)
    inst.meminfo = c.builder.bitcast(ptr_meminfo, inst.meminfo.type)
    inst.data = c.builder.bitcast(ptr_dataptr, inst.data.type)
    ret = inst._getvalue()
    c.context.nrt.incref(c.builder, typ, ret)
    return NativeValue(ret, is_error=c.pyapi.c_api_error())