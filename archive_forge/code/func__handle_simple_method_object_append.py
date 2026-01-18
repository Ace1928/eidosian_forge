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
def _handle_simple_method_object_append(self, node, function, args, is_unbound_method):
    """Optimistic optimisation as X.append() is almost always
        referring to a list.
        """
    if len(args) != 2 or node.result_is_used or node.function.entry:
        return node
    return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyObject_Append', self.PyObject_Append_func_type, args=args, may_return_none=False, is_temp=node.is_temp, result_is_used=False, utility_code=load_c_utility('append'))