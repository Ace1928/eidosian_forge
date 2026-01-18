from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class EvalWithTempExprNode(ExprNodes.ExprNode, LetNodeMixin):
    subexprs = ['temp_expression', 'subexpression']

    def __init__(self, lazy_temp, subexpression):
        self.set_temp_expr(lazy_temp)
        self.pos = subexpression.pos
        self.subexpression = subexpression
        self.type = self.subexpression.type

    def infer_type(self, env):
        return self.subexpression.infer_type(env)

    def may_be_none(self):
        return self.subexpression.may_be_none()

    def result(self):
        return self.subexpression.result()

    def analyse_types(self, env):
        self.temp_expression = self.temp_expression.analyse_types(env)
        self.lazy_temp.update_expression(self.temp_expression)
        self.subexpression = self.subexpression.analyse_types(env)
        self.type = self.subexpression.type
        return self

    def free_subexpr_temps(self, code):
        self.subexpression.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        self.subexpression.generate_disposal_code(code)

    def generate_evaluation_code(self, code):
        self.setup_temp_expr(code)
        self.subexpression.generate_evaluation_code(code)
        self.teardown_temp_expr(code)