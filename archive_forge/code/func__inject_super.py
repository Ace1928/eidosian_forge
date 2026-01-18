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
def _inject_super(self, node, func_name):
    lenv = self.current_env()
    entry = lenv.lookup_here(func_name)
    if entry or node.args:
        return node
    def_node = self.current_scope_node()
    if not isinstance(def_node, Nodes.DefNode) or not def_node.args or len(self.env_stack) < 2:
        return node
    class_node, class_scope = self.env_stack[-2]
    if class_scope.is_py_class_scope:
        def_node.requires_classobj = True
        class_node.class_cell.is_active = True
        node.args = [ExprNodes.ClassCellNode(node.pos, is_generator=def_node.is_generator), ExprNodes.NameNode(node.pos, name=def_node.args[0].name)]
    elif class_scope.is_c_class_scope:
        node.args = [ExprNodes.NameNode(node.pos, name=class_node.scope.name, entry=class_node.entry), ExprNodes.NameNode(node.pos, name=def_node.args[0].name)]
    return node