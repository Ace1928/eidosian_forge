from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _dtype_name(self, dtype):
    name = str(dtype).replace('_', '__').replace(' ', '_')
    if dtype.is_typedef:
        name = Naming.fused_dtype_prefix + name
    return name