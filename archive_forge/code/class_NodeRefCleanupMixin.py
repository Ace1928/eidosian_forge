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
class NodeRefCleanupMixin(object):
    """
    Clean up references to nodes that were replaced.

    NOTE: this implementation assumes that the replacement is
    done first, before hitting any further references during
    normal tree traversal.  This needs to be arranged by calling
    "self.visitchildren()" at a proper place in the transform
    and by ordering the "child_attrs" of nodes appropriately.
    """

    def __init__(self, *args):
        super(NodeRefCleanupMixin, self).__init__(*args)
        self._replacements = {}

    def visit_CloneNode(self, node):
        arg = node.arg
        if arg not in self._replacements:
            self.visitchildren(arg)
        node.arg = self._replacements.get(arg, arg)
        return node

    def visit_ResultRefNode(self, node):
        expr = node.expression
        if expr is None or expr not in self._replacements:
            self.visitchildren(node)
            expr = node.expression
        if expr is not None:
            node.expression = self._replacements.get(expr, expr)
        return node

    def replace(self, node, replacement):
        self._replacements[node] = replacement
        return replacement