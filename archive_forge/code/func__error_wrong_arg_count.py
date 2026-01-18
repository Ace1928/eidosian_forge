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
def _error_wrong_arg_count(self, function_name, node, args, expected=None):
    if not expected:
        arg_str = ''
    elif isinstance(expected, basestring) or expected > 1:
        arg_str = '...'
    elif expected == 1:
        arg_str = 'x'
    else:
        arg_str = ''
    if expected is not None:
        expected_str = 'expected %s, ' % expected
    else:
        expected_str = ''
    error(node.pos, '%s(%s) called with wrong number of args, %sfound %d' % (function_name, arg_str, expected_str, len(args)))