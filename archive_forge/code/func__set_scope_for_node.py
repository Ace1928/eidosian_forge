from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def _set_scope_for_node(self, node, node_scope):
    self._node_scopes[node] = node_scope