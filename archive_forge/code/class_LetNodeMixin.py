from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class LetNodeMixin:

    def set_temp_expr(self, lazy_temp):
        self.lazy_temp = lazy_temp
        self.temp_expression = lazy_temp.expression

    def setup_temp_expr(self, code):
        self.temp_expression.generate_evaluation_code(code)
        self.temp_type = self.temp_expression.type
        if self.temp_type.is_array:
            self.temp_type = c_ptr_type(self.temp_type.base_type)
        self._result_in_temp = self.temp_expression.result_in_temp()
        if self._result_in_temp:
            self.temp = self.temp_expression.result()
        else:
            if self.temp_type.is_memoryviewslice:
                self.temp_expression.make_owned_memoryviewslice(code)
            else:
                self.temp_expression.make_owned_reference(code)
            self.temp = code.funcstate.allocate_temp(self.temp_type, manage_ref=True)
            code.putln('%s = %s;' % (self.temp, self.temp_expression.result()))
            self.temp_expression.generate_disposal_code(code)
            self.temp_expression.free_temps(code)
        self.lazy_temp.result_code = self.temp

    def teardown_temp_expr(self, code):
        if self._result_in_temp:
            self.temp_expression.generate_disposal_code(code)
            self.temp_expression.free_temps(code)
        else:
            if self.temp_type.needs_refcounting:
                code.put_decref_clear(self.temp, self.temp_type)
            code.funcstate.release_temp(self.temp)