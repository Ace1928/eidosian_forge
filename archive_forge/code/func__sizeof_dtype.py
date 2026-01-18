from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _sizeof_dtype(self, dtype):
    if dtype.is_pyobject:
        return 'sizeof(void *)'
    else:
        return 'sizeof(%s)' % self._dtype_type(dtype)