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
def _inject_unicode_predicate(self, node, function, args, is_unbound_method):
    if is_unbound_method or len(args) != 1:
        return node
    ustring = args[0]
    if not isinstance(ustring, ExprNodes.CoerceToPyTypeNode) or not ustring.arg.type.is_unicode_char:
        return node
    uchar = ustring.arg
    method_name = function.attribute
    if method_name == 'istitle':
        utility_code = UtilityCode.load_cached('py_unicode_istitle', 'StringTools.c')
        function_name = '__Pyx_Py_UNICODE_ISTITLE'
    else:
        utility_code = None
        function_name = 'Py_UNICODE_%s' % method_name.upper()
    func_call = self._substitute_method_call(node, function, function_name, self.PyUnicode_uchar_predicate_func_type, method_name, is_unbound_method, [uchar], utility_code=utility_code)
    if node.type.is_pyobject:
        func_call = func_call.coerce_to_pyobject(self.current_env)
    return func_call