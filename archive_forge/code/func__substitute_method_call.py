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
def _substitute_method_call(self, node, function, name, func_type, attr_name, is_unbound_method, args=(), utility_code=None, is_temp=None, may_return_none=ExprNodes.PythonCapiCallNode.may_return_none, with_none_check=True):
    args = list(args)
    if with_none_check and args:
        args[0] = self._wrap_self_arg(args[0], function, is_unbound_method, attr_name)
    if is_temp is None:
        is_temp = node.is_temp
    return ExprNodes.PythonCapiCallNode(node.pos, name, func_type, args=args, is_temp=is_temp, utility_code=utility_code, may_return_none=may_return_none, result_is_used=node.result_is_used)