from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def _transform_tree(self, tree):
    tree.children = list(self._transform_children(tree.children))
    return self._call_userfunc(tree)