from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class HasGilNode(AtomicExprNode):
    """
    Simple node that evaluates to 0 or 1 depending on whether we're
    in a nogil context
    """
    type = c_bint_type

    def analyse_types(self, env):
        return self

    def generate_result_code(self, code):
        self.has_gil = code.funcstate.gil_owned

    def calculate_result_code(self):
        return '1' if self.has_gil else '0'