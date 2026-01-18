from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
class RootScope(Scope):

    def __init__(self, node):
        super(RootScope, self).__init__(None, node)
        self.external_references = {}
        self._parents = {}
        self._nodes_to_names = {}
        self._node_scopes = {}

    def add_external_reference(self, name, node, name_ref=None):
        ref = ExternalReference(name=name, node=node, name_ref=name_ref)
        if name in self.external_references:
            self.external_references[name].append(ref)
        else:
            self.external_references[name] = [ref]

    def get_root_scope(self):
        return self

    def parent(self, node):
        return self._parents.get(node, None)

    def set_parent(self, node, parent):
        self._parents[node] = parent
        if parent is None:
            self._node_scopes[node] = self

    def get_name_for_node(self, node):
        return self._nodes_to_names.get(node, None)

    def set_name_for_node(self, node, name):
        self._nodes_to_names[node] = name

    def lookup_scope(self, node):
        while node:
            try:
                return self._node_scopes[node]
            except KeyError:
                node = self.parent(node)
        return None

    def _set_scope_for_node(self, node, node_scope):
        self._node_scopes[node] = node_scope