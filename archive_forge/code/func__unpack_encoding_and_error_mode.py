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
def _unpack_encoding_and_error_mode(self, pos, args):
    null_node = ExprNodes.NullNode(pos)
    if len(args) >= 2:
        encoding, encoding_node = self._unpack_string_and_cstring_node(args[1])
        if encoding_node is None:
            return None
    else:
        encoding = None
        encoding_node = null_node
    if len(args) == 3:
        error_handling, error_handling_node = self._unpack_string_and_cstring_node(args[2])
        if error_handling_node is None:
            return None
        if error_handling == 'strict':
            error_handling_node = null_node
    else:
        error_handling = 'strict'
        error_handling_node = null_node
    return (encoding, encoding_node, error_handling, error_handling_node)