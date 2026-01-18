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
def _analyse_target_declaration(self, env, is_assignment_expression):
    self.is_target = True
    if not self.entry:
        if is_assignment_expression:
            self.entry = env.lookup_assignment_expression_target(self.name)
        else:
            self.entry = env.lookup_here(self.name)
    if self.entry:
        self.entry.known_standard_library_import = ''
    if not self.entry and self.annotation is not None:
        is_dataclass = env.is_c_dataclass_scope
        self.declare_from_annotation(env, as_target=not is_dataclass)
    elif self.entry and self.entry.is_inherited and self.annotation and env.is_c_dataclass_scope:
        error(self.pos, 'Cannot redeclare inherited fields in Cython dataclasses')
    if not self.entry:
        if env.directives['warn.undeclared']:
            warning(self.pos, "implicit declaration of '%s'" % self.name, 1)
        if env.directives['infer_types'] != False:
            type = unspecified_type
        else:
            type = py_object_type
        if is_assignment_expression:
            self.entry = env.declare_assignment_expression_target(self.name, type, self.pos)
        else:
            self.entry = env.declare_var(self.name, type, self.pos)
    if self.entry.is_declared_generic:
        self.result_ctype = py_object_type
    if self.entry.as_module:
        self.entry.is_variable = 1