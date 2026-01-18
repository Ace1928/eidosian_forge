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
class ComprehensionNode(ScopedExprNode):
    child_attrs = ['loop']
    is_temp = True
    constant_result = not_a_constant

    def infer_type(self, env):
        return self.type

    def analyse_declarations(self, env):
        self.append.target = self
        self.init_scope(env)
        if isinstance(self.loop, Nodes._ForInStatNode):
            assert isinstance(self.loop.iterator, ScopedExprNode), self.loop.iterator
            self.loop.iterator.init_scope(None, env)
        else:
            assert isinstance(self.loop, Nodes.ForFromStatNode), self.loop

    def analyse_scoped_declarations(self, env):
        self.loop.analyse_declarations(env)

    def analyse_types(self, env):
        if not self.has_local_scope:
            self.loop = self.loop.analyse_expressions(env)
        return self

    def analyse_scoped_expressions(self, env):
        if self.has_local_scope:
            self.loop = self.loop.analyse_expressions(env)
        return self

    def may_be_none(self):
        return False

    def generate_result_code(self, code):
        self.generate_operation_code(code)

    def generate_operation_code(self, code):
        if self.type is Builtin.list_type:
            create_code = 'PyList_New(0)'
        elif self.type is Builtin.set_type:
            create_code = 'PySet_New(NULL)'
        elif self.type is Builtin.dict_type:
            create_code = 'PyDict_New()'
        else:
            raise InternalError('illegal type for comprehension: %s' % self.type)
        code.putln('%s = %s; %s' % (self.result(), create_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        self.loop.generate_execution_code(code)

    def annotate(self, code):
        self.loop.annotate(code)