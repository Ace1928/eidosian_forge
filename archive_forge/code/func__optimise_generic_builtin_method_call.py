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
def _optimise_generic_builtin_method_call(self, node, attr_name, function, arg_list, is_unbound_method):
    """
        Try to inject an unbound method call for a call to a method of a known builtin type.
        This enables caching the underlying C function of the method at runtime.
        """
    arg_count = len(arg_list)
    if is_unbound_method or arg_count >= 3 or (not (function.is_attribute and function.is_py_attr)):
        return node
    if not function.obj.type.is_builtin_type:
        return node
    if function.obj.type.name in ('basestring', 'type'):
        return node
    return ExprNodes.CachedBuiltinMethodCallNode(node, function.obj, attr_name, arg_list)