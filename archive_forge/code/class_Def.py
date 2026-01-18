from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
class Def(object):
    """
    Model a definition, either named or unnamed, and its users.
    """
    __slots__ = ('node', '_users')

    def __init__(self, node):
        self.node = node
        self._users = ordered_set()

    def add_user(self, node):
        assert isinstance(node, Def)
        self._users.add(node)

    def name(self):
        """
        If the node associated to this Def has a name, returns this name.
        Otherwise returns its type
        """
        if isinstance(self.node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.node.name
        elif isinstance(self.node, ast.Name):
            return self.node.id
        elif isinstance(self.node, ast.alias):
            base = self.node.name.split('.', 1)[0]
            return self.node.asname or base
        elif isinstance(self.node, tuple):
            return self.node[1]
        else:
            return type(self.node).__name__

    def users(self):
        """
        The list of ast entity that holds a reference to this node
        """
        return self._users

    def __repr__(self):
        return self._repr({})

    def _repr(self, nodes):
        if self in nodes:
            return '(#{})'.format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return '{} -> ({})'.format(self.node, ', '.join((u._repr(nodes.copy()) for u in self._users)))

    def __str__(self):
        return self._str({})

    def _str(self, nodes):
        if self in nodes:
            return '(#{})'.format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return '{} -> ({})'.format(self.name(), ', '.join((u._str(nodes.copy()) for u in self._users)))