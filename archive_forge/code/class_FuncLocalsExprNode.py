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
class FuncLocalsExprNode(DictNode):

    def __init__(self, pos, env):
        local_vars = sorted([entry.name for entry in env.entries.values() if entry.name])
        items = [LocalsDictItemNode(pos, key=IdentifierStringNode(pos, value=var), value=NameNode(pos, name=var, allow_null=True)) for var in local_vars]
        DictNode.__init__(self, pos, key_value_pairs=items, exclude_null_values=True)

    def analyse_types(self, env):
        node = super(FuncLocalsExprNode, self).analyse_types(env)
        node.key_value_pairs = [i for i in node.key_value_pairs if i.value is not None]
        return node