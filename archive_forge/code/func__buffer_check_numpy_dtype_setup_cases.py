from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _buffer_check_numpy_dtype_setup_cases(self, pyx_code):
    """Setup some common cases to match dtypes against specializations"""
    with pyx_code.indenter("if kind in u'iu':"):
        pyx_code.putln('pass')
        pyx_code.named_insertion_point('dtype_int')
    with pyx_code.indenter("elif kind == u'f':"):
        pyx_code.putln('pass')
        pyx_code.named_insertion_point('dtype_float')
    with pyx_code.indenter("elif kind == u'c':"):
        pyx_code.putln('pass')
        pyx_code.named_insertion_point('dtype_complex')
    with pyx_code.indenter("elif kind == u'O':"):
        pyx_code.putln('pass')
        pyx_code.named_insertion_point('dtype_object')