from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _buffer_check_numpy_dtype(self, pyx_code, specialized_buffer_types, pythran_types):
    """
        Match a numpy dtype object to the individual specializations.
        """
    self._buffer_check_numpy_dtype_setup_cases(pyx_code)
    for specialized_type in pythran_types + specialized_buffer_types:
        final_type = specialized_type
        if specialized_type.is_pythran_expr:
            specialized_type = specialized_type.org_buffer
        dtype = specialized_type.dtype
        pyx_code.context.update(itemsize_match=self._sizeof_dtype(dtype) + ' == itemsize', signed_match='not (%s_is_signed ^ dtype_signed)' % self._dtype_name(dtype), dtype=dtype, specialized_type_name=final_type.specialization_string)
        dtypes = [(dtype.is_int, pyx_code.dtype_int), (dtype.is_float, pyx_code.dtype_float), (dtype.is_complex, pyx_code.dtype_complex)]
        for dtype_category, codewriter in dtypes:
            if not dtype_category:
                continue
            cond = '{{itemsize_match}} and (<Py_ssize_t>arg.ndim) == %d' % (specialized_type.ndim,)
            if dtype.is_int:
                cond += ' and {{signed_match}}'
            if final_type.is_pythran_expr:
                cond += ' and arg_is_pythran_compatible'
            with codewriter.indenter('if %s:' % cond):
                codewriter.putln(self.match)
                codewriter.putln('break')