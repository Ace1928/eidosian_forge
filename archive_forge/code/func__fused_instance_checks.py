from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _fused_instance_checks(self, normal_types, pyx_code, env):
    """
        Generate Cython code for instance checks, matching an object to
        specialized types.
        """
    for specialized_type in normal_types:
        py_type_name = specialized_type.py_type_name()
        if py_type_name == 'int':
            py_type_name = '(int, long)'
        pyx_code.context.update(py_type_name=py_type_name, specialized_type_name=specialized_type.specialization_string)
        pyx_code.put_chunk(u"\n                    if isinstance(arg, {{py_type_name}}):\n                        dest_sig[{{dest_sig_idx}}] = '{{specialized_type_name}}'; break\n                ")