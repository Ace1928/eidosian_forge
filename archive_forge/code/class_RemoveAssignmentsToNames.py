from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
class RemoveAssignmentsToNames(VisitorTransform, SkipDeclarations):
    """
    Cython (and Python) normally treats

    class A:
         x = 1

    as generating a class attribute. However for dataclasses the `= 1` should be interpreted as
    a default value to initialize an instance attribute with.
    This transform therefore removes the `x=1` assignment so that the class attribute isn't
    generated, while recording what it has removed so that it can be used in the initialization.
    """

    def __init__(self, names):
        super(RemoveAssignmentsToNames, self).__init__()
        self.names = names
        self.removed_assignments = {}

    def visit_CClassNode(self, node):
        self.visitchildren(node)
        return node

    def visit_PyClassNode(self, node):
        return node

    def visit_FuncDefNode(self, node):
        return node

    def visit_SingleAssignmentNode(self, node):
        if node.lhs.is_name and node.lhs.name in self.names:
            if node.lhs.name in self.removed_assignments:
                warning(node.pos, "Multiple assignments for '%s' in dataclass; using most recent" % node.lhs.name, 1)
            self.removed_assignments[node.lhs.name] = node.rhs
            return []
        return node

    def visit_Node(self, node):
        self.visitchildren(node)
        return node