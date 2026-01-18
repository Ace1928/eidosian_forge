from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _buffer_declarations(self, pyx_code, decl_code, all_buffer_types, pythran_types):
    """
        If we have any buffer specializations, write out some variable
        declarations and imports.
        """
    decl_code.put_chunk(u'\n                ctypedef struct {{memviewslice_cname}}:\n                    void *memview\n\n                void __PYX_XCLEAR_MEMVIEW({{memviewslice_cname}} *, int have_gil)\n                bint __pyx_memoryview_check(object)\n                bint __PYX_IS_PYPY2 "(CYTHON_COMPILING_IN_PYPY && PY_MAJOR_VERSION == 2)"\n            ')
    pyx_code.local_variable_declarations.put_chunk(u'\n                cdef {{memviewslice_cname}} memslice\n                cdef Py_ssize_t itemsize\n                cdef bint dtype_signed\n                cdef Py_UCS4 kind\n\n                itemsize = -1\n            ')
    if pythran_types:
        pyx_code.local_variable_declarations.put_chunk(u'\n                cdef bint arg_is_pythran_compatible\n                cdef Py_ssize_t cur_stride\n            ')
    pyx_code.imports.put_chunk(u'\n                cdef type ndarray\n                ndarray = __Pyx_ImportNumPyArrayTypeIfAvailable()\n            ')
    pyx_code.imports.put_chunk(u'\n                cdef memoryview arg_as_memoryview\n            ')
    seen_typedefs = set()
    seen_int_dtypes = set()
    for buffer_type in all_buffer_types:
        dtype = buffer_type.dtype
        dtype_name = self._dtype_name(dtype)
        if dtype.is_typedef:
            if dtype_name not in seen_typedefs:
                seen_typedefs.add(dtype_name)
                decl_code.putln('ctypedef %s %s "%s"' % (dtype.resolve(), dtype_name, dtype.empty_declaration_code()))
        if buffer_type.dtype.is_int:
            if str(dtype) not in seen_int_dtypes:
                seen_int_dtypes.add(str(dtype))
                pyx_code.context.update(dtype_name=dtype_name, dtype_type=self._dtype_type(dtype))
                pyx_code.local_variable_declarations.put_chunk(u'\n                            cdef bint {{dtype_name}}_is_signed\n                            {{dtype_name}}_is_signed = not (<{{dtype_type}}> -1 > 0)\n                        ')