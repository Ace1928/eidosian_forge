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
class ClassCellNode(ExprNode):
    subexprs = []
    is_temp = True
    is_generator = False
    type = py_object_type

    def analyse_types(self, env):
        return self

    def generate_result_code(self, code):
        if not self.is_generator:
            code.putln('%s = __Pyx_CyFunction_GetClassObj(%s);' % (self.result(), Naming.self_cname))
        else:
            code.putln('%s =  %s->classobj;' % (self.result(), Naming.generator_cname))
        code.putln('if (!%s) { PyErr_SetString(PyExc_SystemError, "super(): empty __class__ cell"); %s }' % (self.result(), code.error_goto(self.pos)))
        code.put_incref(self.result(), py_object_type)