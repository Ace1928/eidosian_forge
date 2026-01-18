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
def _optimise_int_indexing(self, coerce_node, arg, index_node):
    env = self.current_env()
    bound_check_bool = env.directives['boundscheck'] and 1 or 0
    if arg.base.type is Builtin.bytes_type:
        if coerce_node.type in (PyrexTypes.c_char_type, PyrexTypes.c_uchar_type):
            bound_check_node = ExprNodes.IntNode(coerce_node.pos, value=str(bound_check_bool), constant_result=bound_check_bool)
            node = ExprNodes.PythonCapiCallNode(coerce_node.pos, '__Pyx_PyBytes_GetItemInt', self.PyBytes_GetItemInt_func_type, args=[arg.base.as_none_safe_node("'NoneType' object is not subscriptable"), index_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env), bound_check_node], is_temp=True, utility_code=UtilityCode.load_cached('bytes_index', 'StringTools.c'))
            if coerce_node.type is not PyrexTypes.c_char_type:
                node = node.coerce_to(coerce_node.type, env)
            return node
    return coerce_node