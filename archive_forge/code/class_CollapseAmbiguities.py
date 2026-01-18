from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class CollapseAmbiguities(Transformer):
    """
    Transforms a tree that contains any number of _ambig nodes into a list of trees,
    each one containing an unambiguous tree.

    The length of the resulting list is the product of the length of all _ambig nodes.

    Warning: This may quickly explode for highly ambiguous trees.

    """

    def _ambig(self, options):
        return sum(options, [])

    def __default__(self, data, children_lists, meta):
        return [Tree(data, children, meta) for children in combine_alternatives(children_lists)]

    def __default_token__(self, t):
        return [t]