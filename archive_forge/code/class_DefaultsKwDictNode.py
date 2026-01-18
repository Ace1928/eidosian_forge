from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class DefaultsKwDictNode(DictNode):

    def __init__(self, pos, defaults, defaults_struct):
        items = []
        for arg in defaults:
            name = IdentifierStringNode(arg.pos, value=arg.name)
            if not arg.default.is_literal:
                arg = DefaultNonLiteralArgNode(pos, arg, defaults_struct)
            else:
                arg = arg.default
            items.append(DictItemNode(arg.pos, key=name, value=arg))
        super(DefaultsKwDictNode, self).__init__(pos, key_value_pairs=items)