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
def _handle_simple_function_int(self, node, function, pos_args):
    """Transform int() into a faster C function call.
        """
    if len(pos_args) == 0:
        return ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.py_object_type)
    elif len(pos_args) != 1:
        return node
    func_arg = pos_args[0]
    if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
        if func_arg.arg.type.is_float:
            return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyInt_FromDouble', self.PyInt_FromDouble_func_type, args=[func_arg.arg], is_temp=True, py_name='int', utility_code=UtilityCode.load_cached('PyIntFromDouble', 'TypeConversion.c'))
        else:
            return node
    if func_arg.type.is_pyobject and node.type.is_pyobject:
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyNumber_Int', self.PyNumber_Int_func_type, args=pos_args, is_temp=True, py_name='int')
    return node