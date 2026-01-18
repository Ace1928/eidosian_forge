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
def _inject_int_default_argument(self, node, args, arg_index, type, default_value):
    assert len(args) >= arg_index
    if len(args) == arg_index or args[arg_index].is_none:
        args.append(ExprNodes.IntNode(node.pos, value=str(default_value), type=type, constant_result=default_value))
    else:
        arg = args[arg_index].coerce_to(type, self.current_env())
        if isinstance(arg, ExprNodes.CoerceFromPyTypeNode):
            arg.special_none_cvalue = str(default_value)
        args[arg_index] = arg