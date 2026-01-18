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
def _handle_general_function_dict(self, node, pos_args, kwargs):
    """Replace dict(a=b,c=d,...) by the underlying keyword dict
        construction which is done anyway.
        """
    if len(pos_args) > 0:
        return node
    if not isinstance(kwargs, ExprNodes.DictNode):
        return node
    return kwargs