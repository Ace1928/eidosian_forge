from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
class PackedNode(ForestNode):
    """
    A Packed Node represents a single derivation in a symbol node.

    Parameters:
        rule: The rule associated with this node.
        parent: The parent of this node.
        left: The left child of this node. ``None`` if one does not exist.
        right: The right child of this node. ``None`` if one does not exist.
        priority: The priority of this node.
    """
    __slots__ = ('parent', 's', 'rule', 'start', 'left', 'right', 'priority', '_hash')

    def __init__(self, parent, s, rule, start, left, right):
        self.parent = parent
        self.s = s
        self.start = start
        self.rule = rule
        self.left = left
        self.right = right
        self.priority = float('-inf')
        self._hash = hash((self.left, self.right))

    @property
    def is_empty(self):
        return self.left is None and self.right is None

    @property
    def sort_key(self):
        """
        Used to sort PackedNode children of SymbolNodes.
        A SymbolNode has multiple PackedNodes if it matched
        ambiguously. Hence, we use the sort order to identify
        the order in which ambiguous children should be considered.
        """
        return (self.is_empty, -self.priority, self.rule.order)

    @property
    def children(self):
        """Returns a list of this node's children."""
        return [x for x in [self.left, self.right] if x is not None]

    def __iter__(self):
        yield self.left
        yield self.right

    def __eq__(self, other):
        if not isinstance(other, PackedNode):
            return False
        return self is other or (self.left == other.left and self.right == other.right)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        if isinstance(self.s, tuple):
            rule = self.s[0]
            ptr = self.s[1]
            before = (expansion.name for expansion in rule.expansion[:ptr])
            after = (expansion.name for expansion in rule.expansion[ptr:])
            symbol = '{} ::= {}* {}'.format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return '({}, {}, {}, {})'.format(symbol, self.start, self.priority, self.rule.order)