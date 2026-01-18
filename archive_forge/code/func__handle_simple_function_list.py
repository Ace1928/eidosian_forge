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
def _handle_simple_function_list(self, node, function, pos_args):
    """Turn list(ob) into PySequence_List(ob).
        """
    if len(pos_args) != 1:
        return node
    arg = pos_args[0]
    return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PySequence_ListKeepNew' if node.is_temp and arg.is_temp and (arg.type in (PyrexTypes.py_object_type, Builtin.list_type)) else 'PySequence_List', self.PySequence_List_func_type, args=pos_args, is_temp=node.is_temp)