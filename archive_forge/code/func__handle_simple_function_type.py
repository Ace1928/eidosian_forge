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
def _handle_simple_function_type(self, node, function, pos_args):
    """Replace type(o) by a macro call to Py_TYPE(o).
        """
    if len(pos_args) != 1:
        return node
    node = ExprNodes.PythonCapiCallNode(node.pos, 'Py_TYPE', self.Pyx_Type_func_type, args=pos_args, is_temp=False)
    return ExprNodes.CastNode(node, PyrexTypes.py_object_type)