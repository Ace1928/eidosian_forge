from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class TempRefNode(AtomicExprNode):

    def analyse_types(self, env):
        assert self.type == self.handle.type
        return self

    def analyse_target_types(self, env):
        assert self.type == self.handle.type
        return self

    def analyse_target_declaration(self, env):
        pass

    def calculate_result_code(self):
        result = self.handle.temp
        if result is None:
            result = '<error>'
        return result

    def generate_result_code(self, code):
        pass

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        if self.type.is_pyobject:
            rhs.make_owned_reference(code)
            code.put_xdecref(self.result(), self.ctype())
        code.putln('%s = %s;' % (self.result(), rhs.result() if overloaded_assignment else rhs.result_as(self.ctype())))
        rhs.generate_post_assignment_code(code)
        rhs.free_temps(code)