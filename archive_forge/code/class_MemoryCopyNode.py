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
class MemoryCopyNode(ExprNode):
    """
    Wraps a memoryview slice for slice assignment.

        dst: destination mememoryview slice
    """
    subexprs = ['dst']

    def __init__(self, pos, dst):
        super(MemoryCopyNode, self).__init__(pos)
        self.dst = dst
        self.type = dst.type

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        self.dst.generate_evaluation_code(code)
        self._generate_assignment_code(rhs, code)
        self.dst.generate_disposal_code(code)
        self.dst.free_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)