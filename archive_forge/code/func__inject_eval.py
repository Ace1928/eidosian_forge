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
def _inject_eval(self, node, func_name):
    lenv = self.current_env()
    entry = lenv.lookup(func_name)
    if len(node.args) != 1 or (entry and (not entry.is_builtin)):
        return node
    node.args.append(ExprNodes.GlobalsExprNode(node.pos))
    if not lenv.is_module_scope:
        node.args.append(ExprNodes.LocalsExprNode(node.pos, self.current_scope_node(), lenv))
    return node