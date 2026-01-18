from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def _transform_children(self, children):
    for c in children:
        if isinstance(c, Tree):
            res = self._transform_tree(c)
        elif self.__visit_tokens__ and isinstance(c, Token):
            res = self._call_userfunc_token(c)
        else:
            res = c
        if res is not Discard:
            yield res