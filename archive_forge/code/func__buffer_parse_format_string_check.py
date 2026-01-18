from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _buffer_parse_format_string_check(self, pyx_code, decl_code, specialized_type, env):
    """
        For each specialized type, try to coerce the object to a memoryview
        slice of that type. This means obtaining a buffer and parsing the
        format string.
        TODO: separate buffer acquisition from format parsing
        """
    dtype = specialized_type.dtype
    if specialized_type.is_buffer:
        axes = [('direct', 'strided')] * specialized_type.ndim
    else:
        axes = specialized_type.axes
    memslice_type = PyrexTypes.MemoryViewSliceType(dtype, axes)
    memslice_type.create_from_py_utility_code(env)
    pyx_code.context.update(coerce_from_py_func=memslice_type.from_py_function, dtype=dtype)
    decl_code.putln('{{memviewslice_cname}} {{coerce_from_py_func}}(object, int)')
    pyx_code.context.update(specialized_type_name=specialized_type.specialization_string, sizeof_dtype=self._sizeof_dtype(dtype), ndim_dtype=specialized_type.ndim, dtype_is_struct_obj=int(dtype.is_struct or dtype.is_pyobject))
    pyx_code.put_chunk(u"\n                # try {{dtype}}\n                if (((itemsize == -1 and arg_as_memoryview.itemsize == {{sizeof_dtype}})\n                        or itemsize == {{sizeof_dtype}})\n                        and arg_as_memoryview.ndim == {{ndim_dtype}}):\n                    {{if dtype_is_struct_obj}}\n                    if __PYX_IS_PYPY2:\n                        # I wasn't able to diagnose why, but PyPy2 fails to convert a\n                        # memoryview to a Cython memoryview in this case\n                        memslice = {{coerce_from_py_func}}(arg, 0)\n                    else:\n                    {{else}}\n                    if True:\n                    {{endif}}\n                        memslice = {{coerce_from_py_func}}(arg_as_memoryview, 0)\n                    if memslice.memview:\n                        __PYX_XCLEAR_MEMVIEW(&memslice, 1)\n                        # print 'found a match for the buffer through format parsing'\n                        %s\n                        break\n                    else:\n                        __pyx_PyErr_Clear()\n            " % self.match)