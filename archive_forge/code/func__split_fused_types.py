from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _split_fused_types(self, arg):
    """
        Specialize fused types and split into normal types and buffer types.
        """
    specialized_types = PyrexTypes.get_specialized_types(arg.type)
    specialized_types.sort()
    seen_py_type_names = set()
    normal_types, buffer_types, pythran_types = ([], [], [])
    has_object_fallback = False
    for specialized_type in specialized_types:
        py_type_name = specialized_type.py_type_name()
        if py_type_name:
            if py_type_name in seen_py_type_names:
                continue
            seen_py_type_names.add(py_type_name)
            if py_type_name == 'object':
                has_object_fallback = True
            else:
                normal_types.append(specialized_type)
        elif specialized_type.is_pythran_expr:
            pythran_types.append(specialized_type)
        elif specialized_type.is_buffer or specialized_type.is_memoryviewslice:
            buffer_types.append(specialized_type)
    return (normal_types, buffer_types, pythran_types, has_object_fallback)