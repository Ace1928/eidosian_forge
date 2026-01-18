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
def _handle_UnaryMinusNode(self, node):

    def _negate(value):
        if value.startswith('-'):
            value = value[1:]
        else:
            value = '-' + value
        return value
    node_type = node.operand.type
    if isinstance(node.operand, ExprNodes.FloatNode):
        return ExprNodes.FloatNode(node.pos, value=_negate(node.operand.value), type=node_type, constant_result=node.constant_result)
    if node_type.is_int and node_type.signed or (isinstance(node.operand, ExprNodes.IntNode) and node_type.is_pyobject):
        return ExprNodes.IntNode(node.pos, value=_negate(node.operand.value), type=node_type, longness=node.operand.longness, constant_result=node.constant_result)
    return node