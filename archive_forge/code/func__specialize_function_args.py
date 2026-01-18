from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _specialize_function_args(self, args, fused_to_specific):
    for arg in args:
        if arg.type.is_fused:
            arg.type = arg.type.specialize(fused_to_specific)
            if arg.type.is_memoryviewslice:
                arg.type.validate_memslice_dtype(arg.pos)
            if arg.annotation:
                arg.annotation.untyped = True