from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class Transformer_InPlace(Transformer[_Leaf_T, _Return_T]):
    """Same as Transformer, but non-recursive, and changes the tree in-place instead of returning new instances

    Useful for huge trees. Conservative in memory.
    """

    def _transform_tree(self, tree):
        return self._call_userfunc(tree)

    def transform(self, tree: Tree[_Leaf_T]) -> _Return_T:
        for subtree in tree.iter_subtrees():
            subtree.children = list(self._transform_children(subtree.children))
        return self._transform_tree(tree)