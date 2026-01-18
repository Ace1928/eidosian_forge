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
def _handle_simple_method_unicode_count(self, node, function, args, is_unbound_method):
    """Replace unicode.count(...) by a direct call to the
        corresponding C-API function.
        """
    if len(args) not in (2, 3, 4):
        self._error_wrong_arg_count('unicode.count', node, args, '2-4')
        return node
    self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '0')
    self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, 'PY_SSIZE_T_MAX')
    method_call = self._substitute_method_call(node, function, 'PyUnicode_Count', self.PyUnicode_Count_func_type, 'count', is_unbound_method, args)
    return method_call.coerce_to_pyobject(self.current_env())