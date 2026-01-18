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
def _handle_simple_function_tuple(self, node, function, pos_args):
    """Replace tuple([...]) by PyList_AsTuple or PySequence_Tuple.
        """
    if len(pos_args) != 1 or not node.is_temp:
        return node
    arg = pos_args[0]
    if arg.type is Builtin.tuple_type and (not arg.may_be_none()):
        return arg
    if arg.type is Builtin.list_type:
        pos_args[0] = arg.as_none_safe_node("'NoneType' object is not iterable")
        return ExprNodes.PythonCapiCallNode(node.pos, 'PyList_AsTuple', self.PyList_AsTuple_func_type, args=pos_args, is_temp=node.is_temp)
    else:
        return ExprNodes.AsTupleNode(node.pos, arg=arg, type=Builtin.tuple_type)