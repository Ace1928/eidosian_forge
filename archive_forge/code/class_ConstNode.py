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
class ConstNode(AtomicExprNode):
    is_literal = 1
    nogil_check = None

    def is_simple(self):
        return 1

    def nonlocally_immutable(self):
        return 1

    def may_be_none(self):
        return False

    def analyse_types(self, env):
        return self

    def check_const(self):
        return True

    def get_constant_c_result_code(self):
        return self.calculate_result_code()

    def calculate_result_code(self):
        return str(self.value)

    def generate_result_code(self, code):
        pass