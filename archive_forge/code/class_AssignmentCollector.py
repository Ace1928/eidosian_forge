from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class AssignmentCollector(TreeVisitor):

    def __init__(self):
        super(AssignmentCollector, self).__init__()
        self.assignments = []

    def visit_Node(self):
        self._visitchildren(self, None, None)

    def visit_SingleAssignmentNode(self, node):
        self.assignments.append((node.lhs, node.rhs))

    def visit_CascadedAssignmentNode(self, node):
        for lhs in node.lhs_list:
            self.assignments.append((lhs, node.rhs))