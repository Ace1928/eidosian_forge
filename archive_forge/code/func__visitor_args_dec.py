from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def _visitor_args_dec(obj):
    return _apply_v_args(obj, func)