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
class ClassCellInjectorNode(ExprNode):
    is_temp = True
    type = py_object_type
    subexprs = []
    is_active = False

    def analyse_expressions(self, env):
        return self

    def generate_result_code(self, code):
        assert self.is_active
        code.putln('%s = PyList_New(0); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def generate_injection_code(self, code, classobj_cname):
        assert self.is_active
        code.globalstate.use_utility_code(UtilityCode.load_cached('CyFunctionClassCell', 'CythonFunction.c'))
        code.put_error_if_neg(self.pos, '__Pyx_CyFunction_InitClassCell(%s, %s)' % (self.result(), classobj_cname))