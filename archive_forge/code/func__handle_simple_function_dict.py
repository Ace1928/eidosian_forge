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
def _handle_simple_function_dict(self, node, function, pos_args):
    """Replace dict(some_dict) by PyDict_Copy(some_dict).
        """
    if len(pos_args) != 1:
        return node
    arg = pos_args[0]
    if arg.type is Builtin.dict_type:
        arg = arg.as_none_safe_node("'NoneType' is not iterable")
        return ExprNodes.PythonCapiCallNode(node.pos, 'PyDict_Copy', self.PyDict_Copy_func_type, args=[arg], is_temp=node.is_temp)
    return node