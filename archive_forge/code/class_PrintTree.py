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
class PrintTree(TreeVisitor):
    """Prints a representation of the tree to standard output.
    Subclass and override repr_of to provide more information
    about nodes. """

    def __init__(self, start=None, end=None):
        TreeVisitor.__init__(self)
        self._indent = ''
        if start is not None or end is not None:
            self._line_range = (start or 0, end or 2 ** 30)
        else:
            self._line_range = None

    def indent(self):
        self._indent += '  '

    def unindent(self):
        self._indent = self._indent[:-2]

    def __call__(self, tree, phase=None):
        print("Parse tree dump at phase '%s'" % phase)
        self.visit(tree)
        return tree

    def visit_Node(self, node):
        self._print_node(node)
        self.indent()
        self.visitchildren(node)
        self.unindent()
        return node

    def visit_CloneNode(self, node):
        self._print_node(node)
        self.indent()
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            print('%s- %s: %s' % (self._indent, 'arg', self.repr_of(node.arg)))
        self.indent()
        self.visitchildren(node.arg)
        self.unindent()
        self.unindent()
        return node

    def _print_node(self, node):
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            if len(self.access_path) == 0:
                name = '(root)'
            else:
                parent, attr, idx = self.access_path[-1]
                if idx is not None:
                    name = '%s[%d]' % (attr, idx)
                else:
                    name = attr
            print('%s- %s: %s' % (self._indent, name, self.repr_of(node)))

    def repr_of(self, node):
        if node is None:
            return '(none)'
        else:
            result = node.__class__.__name__
            if isinstance(node, ExprNodes.NameNode):
                result += '(type=%s, name="%s")' % (repr(node.type), node.name)
            elif isinstance(node, Nodes.DefNode):
                result += '(name="%s")' % node.name
            elif isinstance(node, Nodes.CFuncDefNode):
                result += '(name="%s")' % node.declared_name()
            elif isinstance(node, ExprNodes.AttributeNode):
                result += '(type=%s, attribute="%s")' % (repr(node.type), node.attribute)
            elif isinstance(node, (ExprNodes.ConstNode, ExprNodes.PyConstNode)):
                result += '(type=%s, value=%r)' % (repr(node.type), node.value)
            elif isinstance(node, ExprNodes.ExprNode):
                t = node.type
                result += '(type=%s)' % repr(t)
            elif node.pos:
                pos = node.pos
                path = pos[0].get_description()
                if '/' in path:
                    path = path.split('/')[-1]
                if '\\' in path:
                    path = path.split('\\')[-1]
                result += '(pos=(%s:%s:%s))' % (path, pos[1], pos[2])
            return result