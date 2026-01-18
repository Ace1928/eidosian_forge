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
class VisitorTransform(TreeVisitor):
    """
    A tree transform is a base class for visitors that wants to do stream
    processing of the structure (rather than attributes etc.) of a tree.

    It implements __call__ to simply visit the argument node.

    It requires the visitor methods to return the nodes which should take
    the place of the visited node in the result tree (which can be the same
    or one or more replacement). Specifically, if the return value from
    a visitor method is:

    - [] or None; the visited node will be removed (set to None if an attribute and
    removed if in a list)
    - A single node; the visited node will be replaced by the returned node.
    - A list of nodes; the visited nodes will be replaced by all the nodes in the
    list. This will only work if the node was already a member of a list; if it
    was not, an exception will be raised. (Typically you want to ensure that you
    are within a StatListNode or similar before doing this.)
    """

    def visitchildren(self, parent, attrs=None, exclude=None):
        return self._process_children(parent, attrs, exclude)

    @cython.final
    def _process_children(self, parent, attrs=None, exclude=None):
        result = self._visitchildren(parent, attrs, exclude)
        for attr, newnode in result.items():
            if type(newnode) is list:
                newnode = self._flatten_list(newnode)
            setattr(parent, attr, newnode)
        return result

    @cython.final
    def _flatten_list(self, orig_list):
        newlist = []
        for x in orig_list:
            if x is not None:
                if type(x) is list:
                    newlist.extend(x)
                else:
                    newlist.append(x)
        return newlist

    def visitchild(self, parent, attr, idx=0):
        child = getattr(parent, attr)
        if child is not None:
            node = self._visitchild(child, parent, attr, idx)
            if node is not child:
                setattr(parent, attr, node)
            child = node
        return child

    def recurse_to_children(self, node):
        self._process_children(node)
        return node

    def __call__(self, root):
        return self._visit(root)