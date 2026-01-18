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
def _handle_simple_method_unicode_split(self, node, function, args, is_unbound_method):
    """Replace unicode.split(...) by a direct call to the
        corresponding C-API function.
        """
    if len(args) not in (1, 2, 3):
        self._error_wrong_arg_count('unicode.split', node, args, '1-3')
        return node
    if len(args) < 2:
        args.append(ExprNodes.NullNode(node.pos))
    else:
        self._inject_null_for_none(args, 1)
    self._inject_int_default_argument(node, args, 2, PyrexTypes.c_py_ssize_t_type, '-1')
    return self._substitute_method_call(node, function, 'PyUnicode_Split', self.PyUnicode_Split_func_type, 'split', is_unbound_method, args)