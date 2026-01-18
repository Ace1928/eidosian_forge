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
class EnvTransform(CythonTransform):
    """
    This transformation keeps a stack of the environments.
    """

    def __call__(self, root):
        self.env_stack = []
        self.enter_scope(root, root.scope)
        return super(EnvTransform, self).__call__(root)

    def current_env(self):
        return self.env_stack[-1][1]

    def current_scope_node(self):
        return self.env_stack[-1][0]

    def global_scope(self):
        return self.current_env().global_scope()

    def enter_scope(self, node, scope):
        self.env_stack.append((node, scope))

    def exit_scope(self):
        self.env_stack.pop()

    def visit_FuncDefNode(self, node):
        self.visit_func_outer_attrs(node)
        self.enter_scope(node, node.local_scope)
        self.visitchildren(node, attrs=None, exclude=node.outer_attrs)
        self.exit_scope()
        return node

    def visit_func_outer_attrs(self, node):
        self.visitchildren(node, attrs=node.outer_attrs)

    def visit_GeneratorBodyDefNode(self, node):
        self._process_children(node)
        return node

    def visit_ClassDefNode(self, node):
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_CStructOrUnionDefNode(self, node):
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_ScopedExprNode(self, node):
        if node.expr_scope:
            self.enter_scope(node, node.expr_scope)
            self._process_children(node)
            self.exit_scope()
        else:
            self._process_children(node)
        return node

    def visit_CArgDeclNode(self, node):
        if node.default:
            attrs = [attr for attr in node.child_attrs if attr != 'default']
            self._process_children(node, attrs)
            self.enter_scope(node, self.current_env().outer_scope)
            self.visitchildren(node, ('default',))
            self.exit_scope()
        else:
            self._process_children(node)
        return node