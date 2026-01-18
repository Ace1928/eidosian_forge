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
def _optimise_numeric_cast_call(self, node, arg):
    function = arg.function
    args = None
    if isinstance(arg, ExprNodes.PythonCapiCallNode):
        args = arg.args
    elif isinstance(function, ExprNodes.NameNode):
        if function.type.is_builtin_type and isinstance(arg.arg_tuple, ExprNodes.TupleNode):
            args = arg.arg_tuple.args
    if args is None or len(args) != 1:
        return node
    func_arg = args[0]
    if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
        func_arg = func_arg.arg
    elif func_arg.type.is_pyobject:
        return node
    if function.name == 'int':
        if func_arg.type.is_int or node.type.is_int:
            if func_arg.type == node.type:
                return func_arg
            elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
                return self._pyucs4_to_number(node, function.name, func_arg)
            elif node.type.assignable_from(func_arg.type) or func_arg.type.is_float:
                return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
        elif func_arg.type.is_float and node.type.is_numeric:
            if func_arg.type.math_h_modifier == 'l':
                truncl = '__Pyx_truncl'
            else:
                truncl = 'trunc' + func_arg.type.math_h_modifier
            return ExprNodes.PythonCapiCallNode(node.pos, truncl, func_type=self.float_float_func_types[func_arg.type], args=[func_arg], py_name='int', is_temp=node.is_temp, result_is_used=node.result_is_used).coerce_to(node.type, self.current_env())
    elif function.name == 'float':
        if func_arg.type.is_float or node.type.is_float:
            if func_arg.type == node.type:
                return func_arg
            elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
                return self._pyucs4_to_number(node, function.name, func_arg)
            elif node.type.assignable_from(func_arg.type) or func_arg.type.is_float:
                return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
    return node