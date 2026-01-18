from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def _handle_simple_function_memoryview(self, node, function, pos_args):
    if len(pos_args) != 1:
        self._error_wrong_arg_count('memoryview', node, pos_args, '1')
        return node
    elif pos_args[0].type.is_pyobject:
        return ExprNodes.PythonCapiCallNode(node.pos, 'PyMemoryView_FromObject', self.PyMemoryView_FromObject_func_type, args=[pos_args[0]], is_temp=node.is_temp, py_name='memoryview')
    elif pos_args[0].type.is_ptr and pos_args[0].base_type is Builtin.py_buffer_type:
        return ExprNodes.PythonCapiCallNode(node.pos, 'PyMemoryView_FromBuffer', self.PyMemoryView_FromBuffer_func_type, args=[pos_args[0]], is_temp=node.is_temp, py_name='memoryview')
    return node