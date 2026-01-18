from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class Visitor_Recursive(VisitorBase, Generic[_Leaf_T]):
    """Bottom-up visitor, recursive.

    Visiting a node calls its methods (provided by the user via inheritance) according to ``tree.data``

    Slightly faster than the non-recursive version.
    """

    def visit(self, tree: Tree[_Leaf_T]) -> Tree[_Leaf_T]:
        """Visits the tree, starting with the leaves and finally the root (bottom-up)"""
        for child in tree.children:
            if isinstance(child, Tree):
                self.visit(child)
        self._call_userfunc(tree)
        return tree

    def visit_topdown(self, tree: Tree[_Leaf_T]) -> Tree[_Leaf_T]:
        """Visit the tree, starting at the root, and ending at the leaves (top-down)"""
        self._call_userfunc(tree)
        for child in tree.children:
            if isinstance(child, Tree):
                self.visit_topdown(child)
        return tree