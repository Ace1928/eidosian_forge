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
def _handle_simple_method_dict_setdefault(self, node, function, args, is_unbound_method):
    """Replace dict.setdefault() by calls to PyDict_GetItem() and PyDict_SetItem().
        """
    if len(args) == 2:
        args.append(ExprNodes.NoneNode(node.pos))
    elif len(args) != 3:
        self._error_wrong_arg_count('dict.setdefault', node, args, '2 or 3')
        return node
    key_type = args[1].type
    if key_type.is_builtin_type:
        is_safe_type = int(key_type.name in 'str bytes unicode float int long bool')
    elif key_type is PyrexTypes.py_object_type:
        is_safe_type = -1
    else:
        is_safe_type = 0
    args.append(ExprNodes.IntNode(node.pos, value=str(is_safe_type), constant_result=is_safe_type))
    return self._substitute_method_call(node, function, '__Pyx_PyDict_SetDefault', self.Pyx_PyDict_SetDefault_func_type, 'setdefault', is_unbound_method, args, may_return_none=True, utility_code=load_c_utility('dict_setdefault'))