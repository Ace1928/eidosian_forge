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
def _handle_simple_function_slice(self, node, pos_args):
    arg_count = len(pos_args)
    start = step = None
    if arg_count == 1:
        stop, = pos_args
    elif arg_count == 2:
        start, stop = pos_args
    elif arg_count == 3:
        start, stop, step = pos_args
    else:
        self._error_wrong_arg_count('slice', node, pos_args)
        return node
    return ExprNodes.SliceNode(node.pos, start=start or ExprNodes.NoneNode(node.pos), stop=stop, step=step or ExprNodes.NoneNode(node.pos))