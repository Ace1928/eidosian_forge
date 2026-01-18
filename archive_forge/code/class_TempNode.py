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
class TempNode(ExprNode):
    subexprs = []

    def __init__(self, pos, type, env=None):
        ExprNode.__init__(self, pos)
        self.type = type
        if type.is_pyobject:
            self.result_ctype = py_object_type
        self.is_temp = 1

    def analyse_types(self, env):
        return self

    def analyse_target_declaration(self, env):
        self.is_target = True

    def generate_result_code(self, code):
        pass

    def allocate(self, code):
        self.temp_cname = code.funcstate.allocate_temp(self.type, manage_ref=True)

    def release(self, code):
        code.funcstate.release_temp(self.temp_cname)
        self.temp_cname = None

    def result(self):
        try:
            return self.temp_cname
        except:
            assert False, 'Remember to call allocate/release on TempNode'
            raise

    def allocate_temp_result(self, code):
        pass

    def release_temp_result(self, code):
        pass