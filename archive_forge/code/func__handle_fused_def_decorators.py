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
def _handle_fused_def_decorators(self, old_decorators, env, node):
    """
        Create function calls to the decorators and reassignments to
        the function.
        """
    decorators = []
    for decorator in old_decorators:
        func = decorator.decorator
        if not func.is_name or func.name not in ('staticmethod', 'classmethod') or env.lookup_here(func.name):
            decorators.append(decorator)
    if decorators:
        transform = DecoratorTransform(self.context)
        def_node = node.node
        _, reassignments = transform.chain_decorators(def_node, decorators, def_node.name)
        reassignments.analyse_declarations(env)
        node = [node, reassignments]
    return node