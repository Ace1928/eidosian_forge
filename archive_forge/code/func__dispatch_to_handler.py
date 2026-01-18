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
def _dispatch_to_handler(self, node, function, args, kwargs=None):
    if kwargs is None:
        handler_name = '_handle_simple_function_%s' % function.name
    else:
        handler_name = '_handle_general_function_%s' % function.name
    handle_call = getattr(self, handler_name, None)
    if handle_call is not None:
        if kwargs is None:
            return handle_call(node, args)
        else:
            return handle_call(node, args, kwargs)
    return node