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
class CppIteratorNode(ExprNode):
    cpp_sequence_cname = None
    cpp_attribute_op = '.'
    extra_dereference = ''
    is_temp = True
    reversed = False
    subexprs = ['sequence']

    def get_iterator_func_names(self):
        return ('begin', 'end') if not self.reversed else ('rbegin', 'rend')

    def analyse_types(self, env):
        sequence_type = self.sequence.type
        if sequence_type.is_ptr:
            sequence_type = sequence_type.base_type
        begin_name, end_name = self.get_iterator_func_names()
        begin = sequence_type.scope.lookup(begin_name)
        end = sequence_type.scope.lookup(end_name)
        if begin is None or not begin.type.is_cfunction or begin.type.args:
            error(self.pos, 'missing %s() on %s' % (begin_name, self.sequence.type))
            self.type = error_type
            return self
        if end is None or not end.type.is_cfunction or end.type.args:
            error(self.pos, 'missing %s() on %s' % (end_name, self.sequence.type))
            self.type = error_type
            return self
        iter_type = begin.type.return_type
        if iter_type.is_cpp_class:
            if env.directives['cpp_locals']:
                self.extra_dereference = '*'
            if env.lookup_operator_for_types(self.pos, '!=', [iter_type, end.type.return_type]) is None:
                error(self.pos, 'missing operator!= on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            if env.lookup_operator_for_types(self.pos, '++', [iter_type]) is None:
                error(self.pos, 'missing operator++ on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            if env.lookup_operator_for_types(self.pos, '*', [iter_type]) is None:
                error(self.pos, 'missing operator* on result of %s() on %s' % (begin_name, self.sequence.type))
                self.type = error_type
                return self
            self.type = iter_type
        elif iter_type.is_ptr:
            if not iter_type == end.type.return_type:
                error(self.pos, 'incompatible types for %s() and %s()' % (begin_name, end_name))
            self.type = iter_type
        else:
            error(self.pos, 'result type of %s() on %s must be a C++ class or pointer' % (begin_name, self.sequence.type))
            self.type = error_type
        return self

    def generate_result_code(self, code):
        sequence_type = self.sequence.type
        begin_name, _ = self.get_iterator_func_names()
        if self.sequence.is_simple():
            code.putln('%s = %s%s%s();' % (self.result(), self.sequence.result(), self.cpp_attribute_op, begin_name))
        else:
            temp_type = sequence_type
            if temp_type.is_reference:
                temp_type = PyrexTypes.CPtrType(sequence_type.ref_base_type)
            if temp_type.is_ptr or code.globalstate.directives['cpp_locals']:
                self.cpp_attribute_op = '->'
            self.cpp_sequence_cname = code.funcstate.allocate_temp(temp_type, manage_ref=False)
            code.putln('%s = %s%s;' % (self.cpp_sequence_cname, '&' if temp_type.is_ptr else '', self.sequence.move_result_rhs()))
            code.putln('%s = %s%s%s();' % (self.result(), self.cpp_sequence_cname, self.cpp_attribute_op, begin_name))

    def generate_iter_next_result_code(self, result_name, code):
        _, end_name = self.get_iterator_func_names()
        code.putln('if (!(%s%s != %s%s%s())) break;' % (self.extra_dereference, self.result(), self.cpp_sequence_cname or self.sequence.result(), self.cpp_attribute_op, end_name))
        code.putln('%s = *%s%s;' % (result_name, self.extra_dereference, self.result()))
        code.putln('++%s%s;' % (self.extra_dereference, self.result()))

    def generate_subexpr_disposal_code(self, code):
        if not self.cpp_sequence_cname:
            return
        ExprNode.generate_subexpr_disposal_code(self, code)

    def free_subexpr_temps(self, code):
        if not self.cpp_sequence_cname:
            return
        ExprNode.free_subexpr_temps(self, code)

    def generate_disposal_code(self, code):
        if not self.cpp_sequence_cname:
            ExprNode.generate_subexpr_disposal_code(self, code)
            ExprNode.free_subexpr_temps(self, code)
        ExprNode.generate_disposal_code(self, code)

    def free_temps(self, code):
        if self.cpp_sequence_cname:
            code.funcstate.release_temp(self.cpp_sequence_cname)
        ExprNode.free_temps(self, code)