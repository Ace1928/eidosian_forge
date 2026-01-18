from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class CoerceCppTemps(EnvTransform, SkipDeclarations):
    """
    For temporary expression that are implemented using std::optional it's necessary the temps are
    assigned using `__pyx_t_x = value;` but accessed using `something = (*__pyx_t_x)`. This transform
    inserts a coercion node to take care of this, and runs absolutely last (once nothing else can be
    inserted into the tree)

    TODO: a possible alternative would be to split ExprNode.result() into ExprNode.rhs_rhs() and ExprNode.lhs_rhs()???
    """

    def visit_ModuleNode(self, node):
        if self.current_env().cpp:
            self.visitchildren(node)
        return node

    def visit_ExprNode(self, node):
        self.visitchildren(node)
        if self.current_env().directives['cpp_locals'] and node.is_temp and node.type.is_cpp_class and (not node.type.is_fake_reference):
            node = ExprNodes.CppOptionalTempCoercion(node)
        return node