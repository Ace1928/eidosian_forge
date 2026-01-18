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
class PyClassMetaclassNode(ExprNode):
    subexprs = []

    def analyse_types(self, env):
        self.type = py_object_type
        self.is_temp = True
        return self

    def may_be_none(self):
        return True

    def generate_result_code(self, code):
        bases = self.class_def_node.bases
        mkw = self.class_def_node.mkw
        if mkw:
            code.globalstate.use_utility_code(UtilityCode.load_cached('Py3MetaclassGet', 'ObjectHandling.c'))
            call = '__Pyx_Py3MetaclassGet(%s, %s)' % (bases.result(), mkw.result())
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('CalculateMetaclass', 'ObjectHandling.c'))
            call = '__Pyx_CalculateMetaclass(NULL, %s)' % bases.result()
        code.putln('%s = %s; %s' % (self.result(), call, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)