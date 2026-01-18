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
def _handle_simple_function_str(self, node, function, pos_args):
    """Optimize single argument calls to str().
        """
    if len(pos_args) != 1:
        if len(pos_args) == 0:
            return ExprNodes.StringNode(node.pos, value=EncodedString(), constant_result='')
        return node
    arg = pos_args[0]
    if arg.type is Builtin.str_type:
        if not arg.may_be_none():
            return arg
        cname = '__Pyx_PyStr_Str'
        utility_code = UtilityCode.load_cached('PyStr_Str', 'StringTools.c')
    else:
        cname = '__Pyx_PyObject_Str'
        utility_code = UtilityCode.load_cached('PyObject_Str', 'StringTools.c')
    return ExprNodes.PythonCapiCallNode(node.pos, cname, self.PyObject_String_func_type, args=pos_args, is_temp=node.is_temp, utility_code=utility_code, py_name='str')