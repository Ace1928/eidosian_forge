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
def _handle_def(self, decorators, env, node):
    """Handle def or cpdef fused functions"""
    node.stats.insert(0, node.py_func)
    self.visitchild(node, 'py_func')
    node.update_fused_defnode_entry(env)
    node.py_func.entry.signature.use_fastcall = False
    pycfunc = ExprNodes.PyCFunctionNode.from_defnode(node.py_func, binding=True)
    pycfunc = ExprNodes.ProxyNode(pycfunc.coerce_to_temp(env))
    node.resulting_fused_function = pycfunc
    node.fused_func_assignment = self._create_assignment(node.py_func, ExprNodes.CloneNode(pycfunc), env)
    if decorators:
        node = self._handle_fused_def_decorators(decorators, env, node)
    return node