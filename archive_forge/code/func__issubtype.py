import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def _issubtype(left, right):
    if _contains_unbound_typevar(left):
        return True
    if right is None:
        return True
    if _contains_unbound_typevar(right):
        return True
    try:
        return issubtype(left, right)
    except TypeError:
        return True