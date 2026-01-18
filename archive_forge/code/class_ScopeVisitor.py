from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
class ScopeVisitor(ast.NodeVisitor):

    def __init__(self):
        super(ScopeVisitor, self).__init__()
        self._parent = None
        self.root_scope = self.scope = RootScope(None)

    def visit(self, node):
        if node is None:
            return
        if self.root_scope.node is None:
            self.root_scope.node = node
        self.root_scope.set_parent(node, self._parent)
        tmp = self._parent
        self._parent = node
        super(ScopeVisitor, self).visit(node)
        self._parent = tmp

    def visit_in_order(self, node, *attrs):
        for attr in attrs:
            val = getattr(node, attr, None)
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    self.visit(item)
            elif isinstance(val, ast.AST):
                self.visit(val)

    def visit_Import(self, node):
        for alias in node.names:
            name_parts = alias.name.split('.')
            if not alias.asname:
                cur_name = self.scope.define_name(name_parts[0], alias)
                self.root_scope.add_external_reference(name_parts[0], alias, name_ref=cur_name)
                partial_name = name_parts[0]
                for part in name_parts[1:]:
                    partial_name += '.' + part
                    cur_name = cur_name.lookup_name(part)
                    cur_name.define(alias)
                    self.root_scope.add_external_reference(partial_name, alias, name_ref=cur_name)
            else:
                name = self.scope.define_name(alias.asname, alias)
                for i in range(1, len(name_parts)):
                    self.root_scope.add_external_reference('.'.join(name_parts[:i]), alias)
                self.root_scope.add_external_reference(alias.name, alias, name_ref=name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            name_parts = node.module.split('.')
            for i in range(1, len(name_parts) + 1):
                self.root_scope.add_external_reference('.'.join(name_parts[:i]), node)
        for alias in node.names:
            name = self.scope.define_name(alias.asname or alias.name, alias)
            if node.module:
                self.root_scope.add_external_reference('.'.join((node.module, alias.name)), alias, name_ref=name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self.scope.define_name(node.id, node)
        elif isinstance(node.ctx, ast.Load):
            self.scope.lookup_name(node.id).add_reference(node)
            self.root_scope.set_name_for_node(node, self.scope.lookup_name(node.id))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.visit_in_order(node, 'decorator_list')
        if isinstance(self.root_scope.parent(node), ast.ClassDef):
            pass
        else:
            self.scope.define_name(node.name, node)
        try:
            self.scope = self.scope.create_scope(node)
            self.visit_in_order(node, 'args', 'returns', 'body')
        finally:
            self.scope = self.scope.parent_scope

    def visit_arguments(self, node):
        self.visit_in_order(node, 'defaults', 'args')
        if six.PY2:
            for arg_attr_name in ('vararg', 'kwarg'):
                arg_name = getattr(node, arg_attr_name, None)
                if arg_name is not None:
                    self.scope.define_name(arg_name, node)
        else:
            self.visit_in_order(node, 'vararg', 'kwarg')

    def visit_arg(self, node):
        self.scope.define_name(node.arg, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.visit_in_order(node, 'decorator_list', 'bases')
        self.scope.define_name(node.name, node)
        try:
            self.scope = self.scope.create_scope(node)
            self.visit_in_order(node, 'body')
        finally:
            self.scope = self.scope.parent_scope

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node_value_name = self.root_scope.get_name_for_node(node.value)
        if node_value_name:
            node_name = node_value_name.lookup_name(node.attr)
            self.root_scope.set_name_for_node(node, node_name)
            node_name.add_reference(node)