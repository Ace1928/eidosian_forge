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
class FindInvalidUseOfFusedTypes(CythonTransform):

    def visit_FuncDefNode(self, node):
        if not node.has_fused_arguments:
            if not node.is_generator_body and node.return_type.is_fused:
                error(node.pos, 'Return type is not specified as argument type')
            else:
                self.visitchildren(node)
        return node

    def visit_ExprNode(self, node):
        if node.type and node.type.is_fused:
            error(node.pos, 'Invalid use of fused types, type cannot be specialized')
        else:
            self.visitchildren(node)
        return node