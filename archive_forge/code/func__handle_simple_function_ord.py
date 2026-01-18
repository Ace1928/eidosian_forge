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
def _handle_simple_function_ord(self, node, function, pos_args):
    """Unpack ord(Py_UNICODE) and ord('X').
        """
    if len(pos_args) != 1:
        return node
    arg = pos_args[0]
    if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
        if arg.arg.type.is_unicode_char:
            return ExprNodes.TypecastNode(arg.pos, operand=arg.arg, type=PyrexTypes.c_long_type).coerce_to(node.type, self.current_env())
    elif isinstance(arg, ExprNodes.UnicodeNode):
        if len(arg.value) == 1:
            return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_int_type, value=str(ord(arg.value)), constant_result=ord(arg.value)).coerce_to(node.type, self.current_env())
    elif isinstance(arg, ExprNodes.StringNode):
        if arg.unicode_value and len(arg.unicode_value) == 1 and (ord(arg.unicode_value) <= 255):
            return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_int_type, value=str(ord(arg.unicode_value)), constant_result=ord(arg.unicode_value)).coerce_to(node.type, self.current_env())
    return node