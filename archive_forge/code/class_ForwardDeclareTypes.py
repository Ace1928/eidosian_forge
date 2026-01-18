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
class ForwardDeclareTypes(CythonTransform):
    """
    Declare all global cdef names that we allow referencing in other places,
    before declaring everything (else) in source code order.
    """

    def visit_CompilerDirectivesNode(self, node):
        env = self.module_scope
        old = env.directives
        env.directives = node.directives
        self.visitchildren(node)
        env.directives = old
        return node

    def visit_ModuleNode(self, node):
        self.module_scope = node.scope
        self.module_scope.directives = node.directives
        self.visitchildren(node)
        return node

    def visit_CDefExternNode(self, node):
        old_cinclude_flag = self.module_scope.in_cinclude
        self.module_scope.in_cinclude = 1
        self.visitchildren(node)
        self.module_scope.in_cinclude = old_cinclude_flag
        return node

    def visit_CEnumDefNode(self, node):
        node.declare(self.module_scope)
        return node

    def visit_CStructOrUnionDefNode(self, node):
        if node.name not in self.module_scope.entries:
            node.declare(self.module_scope)
        return node

    def visit_CClassDefNode(self, node):
        if node.class_name not in self.module_scope.entries:
            node.declare(self.module_scope)
        type = self.module_scope.entries[node.class_name].type
        if type is not None and type.is_extension_type and (not type.is_builtin_type) and type.scope:
            scope = type.scope
            for entry in scope.cfunc_entries:
                if entry.type and entry.type.is_fused:
                    entry.type.get_all_specialized_function_types()
        return node

    def visit_FuncDefNode(self, node):
        return node

    def visit_PyClassDefNode(self, node):
        return node