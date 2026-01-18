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
def _DISABLED_handle_simple_function_tuple(self, node, pos_args):
    if not pos_args:
        return ExprNodes.TupleNode(node.pos, args=[], constant_result=())
    result = self._transform_list_set_genexpr(node, pos_args, Builtin.list_type)
    if result is not node:
        return ExprNodes.AsTupleNode(node.pos, arg=result)
    return node