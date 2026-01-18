from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def _visit_tree(self, tree: Tree[_Leaf_T]):
    f = getattr(self, tree.data)
    wrapper = getattr(f, 'visit_wrapper', None)
    if wrapper is not None:
        return f.visit_wrapper(f, tree.data, tree.children, tree.meta)
    else:
        return f(tree)