import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def _contains_unbound_typevar(t: Type) -> bool:
    """Recursively check if `t` or any types contained by `t` is a `TypeVar`.

    Examples where we return `True`: `T`, `Optional[T]`, `Tuple[Optional[T], ...]`, ...
    Examples where we return `False`: `int`, `Optional[str]`, ...

    :param t: Type to evaluate.
    :return: `True` if the input type contains an unbound `TypeVar`, `False` otherwise.
    """
    if isinstance(t, TypeVar):
        return True
    for arg in get_args(t):
        if _contains_unbound_typevar(arg):
            return True
    return False