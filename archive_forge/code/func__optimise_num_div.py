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
def _optimise_num_div(self, operator, node, function, args, is_unbound_method):
    if len(args) != 2 or not args[1].has_constant_result() or args[1].constant_result == 0:
        return node
    if isinstance(args[1], ExprNodes.IntNode):
        if not -2 ** 30 <= args[1].constant_result <= 2 ** 30:
            return node
    elif isinstance(args[1], ExprNodes.FloatNode):
        if not -2 ** 53 <= args[1].constant_result <= 2 ** 53:
            return node
    else:
        return node
    return self._optimise_num_binop(operator, node, function, args, is_unbound_method)