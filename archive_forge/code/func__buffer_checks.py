from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _buffer_checks(self, buffer_types, pythran_types, pyx_code, decl_code, accept_none, env):
    """
        Generate Cython code to match objects to buffer specializations.
        First try to get a numpy dtype object and match it against the individual
        specializations. If that fails, try naively to coerce the object
        to each specialization, which obtains the buffer each time and tries
        to match the format string.
        """
    pyx_code.put_chunk(u'\n                ' + (u'arg_is_pythran_compatible = False' if pythran_types else u'') + u'\n                if ndarray is not None:\n                    if isinstance(arg, ndarray):\n                        dtype = arg.dtype\n                        ' + (u'arg_is_pythran_compatible = True' if pythran_types else u'') + u"\n                    elif __pyx_memoryview_check(arg):\n                        arg_base = arg.base\n                        if isinstance(arg_base, ndarray):\n                            dtype = arg_base.dtype\n                        else:\n                            dtype = None\n                    else:\n                        dtype = None\n\n                    itemsize = -1\n                    if dtype is not None:\n                        itemsize = dtype.itemsize\n                        kind = ord(dtype.kind)\n                        dtype_signed = kind == u'i'\n            ")
    pyx_code.indent(2)
    if pythran_types:
        pyx_code.put_chunk(u'\n                        # Pythran only supports the endianness of the current compiler\n                        byteorder = dtype.byteorder\n                        if byteorder == "<" and not __Pyx_Is_Little_Endian():\n                            arg_is_pythran_compatible = False\n                        elif byteorder == ">" and __Pyx_Is_Little_Endian():\n                            arg_is_pythran_compatible = False\n                        if arg_is_pythran_compatible:\n                            cur_stride = itemsize\n                            shape = arg.shape\n                            strides = arg.strides\n                            for i in range(arg.ndim-1, -1, -1):\n                                if (<Py_ssize_t>strides[i]) != cur_stride:\n                                    arg_is_pythran_compatible = False\n                                    break\n                                cur_stride *= <Py_ssize_t> shape[i]\n                            else:\n                                arg_is_pythran_compatible = not (arg.flags.f_contiguous and (<Py_ssize_t>arg.ndim) > 1)\n                ')
    pyx_code.named_insertion_point('numpy_dtype_checks')
    self._buffer_check_numpy_dtype(pyx_code, buffer_types, pythran_types)
    pyx_code.dedent(2)
    if accept_none:
        pyx_code.context.update(specialized_type_name=buffer_types[0].specialization_string)
        pyx_code.put_chunk('\n                if arg is None:\n                    %s\n                    break\n                ' % self.match)
    pyx_code.put_chunk('\n            try:\n                arg_as_memoryview = memoryview(arg)\n            except (ValueError, TypeError):\n                pass\n            ')
    with pyx_code.indenter('else:'):
        for specialized_type in buffer_types:
            self._buffer_parse_format_string_check(pyx_code, decl_code, specialized_type, env)