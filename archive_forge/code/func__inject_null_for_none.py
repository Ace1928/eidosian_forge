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
def _inject_null_for_none(self, args, index):
    if len(args) <= index:
        return
    arg = args[index]
    args[index] = ExprNodes.NullNode(arg.pos) if arg.is_none else ExprNodes.PythonCapiCallNode(arg.pos, '__Pyx_NoneAsNull', self.obj_to_obj_func_type, args=[arg.coerce_to_simple(self.current_env())], is_temp=0)