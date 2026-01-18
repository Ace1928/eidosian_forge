from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
class ScopeTrackingTransform(CythonTransform):

    def visit_ModuleNode(self, node):
        self.scope_type = 'module'
        self.scope_node = node
        self._process_children(node)
        return node

    def visit_scope(self, node, scope_type):
        prev = (self.scope_type, self.scope_node)
        self.scope_type = scope_type
        self.scope_node = node
        self._process_children(node)
        self.scope_type, self.scope_node = prev
        return node

    def visit_CClassDefNode(self, node):
        return self.visit_scope(node, 'cclass')

    def visit_PyClassDefNode(self, node):
        return self.visit_scope(node, 'pyclass')

    def visit_FuncDefNode(self, node):
        return self.visit_scope(node, 'function')

    def visit_CStructOrUnionDefNode(self, node):
        return self.visit_scope(node, 'struct')