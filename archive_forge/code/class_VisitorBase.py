from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class VisitorBase:

    def _call_userfunc(self, tree):
        return getattr(self, tree.data, self.__default__)(tree)

    def __default__(self, tree):
        """Default function that is called if there is no attribute matching ``tree.data``

        Can be overridden. Defaults to doing nothing.
        """
        return tree

    def __class_getitem__(cls, _):
        return cls