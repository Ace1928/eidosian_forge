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
def _create_merge_node_if_necessary(self, env):
    self._flatten_starred_args()
    if not any((arg.is_starred for arg in self.args)):
        return self
    args = []
    values = []
    for arg in self.args:
        if arg.is_starred:
            if values:
                args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
                values = []
            args.append(arg.target)
        else:
            values.append(arg)
    if values:
        args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
    node = MergedSequenceNode(self.pos, args, self.type)
    if self.mult_factor:
        node = binop_node(self.pos, '*', node, self.mult_factor.coerce_to_pyobject(env), inplace=True, type=self.type, is_temp=True)
    return node