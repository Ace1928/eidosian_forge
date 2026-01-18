from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def _python_set_to_native(typ, obj, c, size, setptr, errorptr):
    """
    Construct a new native set from a Python set.
    """
    ok, inst = setobj.SetInstance.allocate_ex(c.context, c.builder, typ, size)
    with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
        with if_ok:
            typobjptr = cgutils.alloca_once_value(c.builder, ir.Constant(c.pyapi.pyobj, None))
            with c.pyapi.set_iterate(obj) as loop:
                itemobj = loop.value
                typobj = c.pyapi.get_type(itemobj)
                expected_typobj = c.builder.load(typobjptr)
                with c.builder.if_else(cgutils.is_null(c.builder, expected_typobj), likely=False) as (if_first, if_not_first):
                    with if_first:
                        c.builder.store(typobj, typobjptr)
                    with if_not_first:
                        type_mismatch = c.builder.icmp_signed('!=', typobj, expected_typobj)
                        with c.builder.if_then(type_mismatch, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                            c.pyapi.err_set_string('PyExc_TypeError', "can't unbox heterogeneous set")
                            loop.do_break()
                native = c.unbox(typ.dtype, itemobj)
                with c.builder.if_then(native.is_error, likely=False):
                    c.builder.store(cgutils.true_bit, errorptr)
                inst.add_pyapi(c.pyapi, native.value, do_resize=False)
            if typ.reflected:
                inst.parent = obj
            with c.builder.if_then(c.builder.not_(c.builder.load(errorptr)), likely=False):
                c.pyapi.object_set_private_data(obj, inst.meminfo)
            inst.set_dirty(False)
            c.builder.store(inst.value, setptr)
        with if_not_ok:
            c.builder.store(cgutils.true_bit, errorptr)
    with c.builder.if_then(c.builder.load(errorptr)):
        c.context.nrt.decref(c.builder, typ, inst.value)