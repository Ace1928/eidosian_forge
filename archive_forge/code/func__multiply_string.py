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
def _multiply_string(self, node, string_node, multiplier_node):
    multiplier = multiplier_node.constant_result
    if not isinstance(multiplier, _py_int_types):
        return node
    if not (node.has_constant_result() and isinstance(node.constant_result, _py_string_types)):
        return node
    if len(node.constant_result) > 256:
        return node
    build_string = encoded_string
    if isinstance(string_node, ExprNodes.BytesNode):
        build_string = bytes_literal
    elif isinstance(string_node, ExprNodes.StringNode):
        if string_node.unicode_value is not None:
            string_node.unicode_value = encoded_string(string_node.unicode_value * multiplier, string_node.unicode_value.encoding)
        build_string = encoded_string if string_node.value.is_unicode else bytes_literal
    elif isinstance(string_node, ExprNodes.UnicodeNode):
        if string_node.bytes_value is not None:
            string_node.bytes_value = bytes_literal(string_node.bytes_value * multiplier, string_node.bytes_value.encoding)
    else:
        assert False, 'unknown string node type: %s' % type(string_node)
    string_node.value = build_string(string_node.value * multiplier, string_node.value.encoding)
    if isinstance(string_node, ExprNodes.StringNode) and string_node.unicode_value is not None:
        string_node.constant_result = string_node.unicode_value
    else:
        string_node.constant_result = string_node.value
    return string_node