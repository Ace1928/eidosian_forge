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
def _handle_simple_method_unicode_replace(self, node, function, args, is_unbound_method):
    """Replace unicode.replace(...) by a direct call to the
        corresponding C-API function.
        """
    if len(args) not in (3, 4):
        self._error_wrong_arg_count('unicode.replace', node, args, '3-4')
        return node
    self._inject_int_default_argument(node, args, 3, PyrexTypes.c_py_ssize_t_type, '-1')
    return self._substitute_method_call(node, function, 'PyUnicode_Replace', self.PyUnicode_Replace_func_type, 'replace', is_unbound_method, args)