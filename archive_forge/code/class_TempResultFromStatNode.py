from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class TempResultFromStatNode(ExprNodes.ExprNode):
    subexprs = []
    child_attrs = ['body']

    def __init__(self, result_ref, body):
        self.result_ref = result_ref
        self.pos = body.pos
        self.body = body
        self.type = result_ref.type
        self.is_temp = 1

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)

    def analyse_types(self, env):
        self.body = self.body.analyse_expressions(env)
        return self

    def may_be_none(self):
        return self.result_ref.may_be_none()

    def generate_result_code(self, code):
        self.result_ref.result_code = self.result()
        self.body.generate_execution_code(code)

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)