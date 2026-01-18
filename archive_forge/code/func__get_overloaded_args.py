import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
def _get_overloaded_args(relevant_args: Iterable[Any], get_type_fn: Callable[[Any], Type]=None) -> List[Any]:
    """Returns a list of arguments on which to call __torch_function__.

    Checks arguments in relevant_args for __torch_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    get_type_fn : callable, optional
        Function to call on each argument in relevant_args to get its type.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    if get_type_fn is None:
        get_type_fn = type
    if not torch._C._is_torch_function_enabled():
        return []
    overloaded_types: Set[Type] = set()
    overloaded_args: List[Any] = []
    for arg in relevant_args:
        arg_type = get_type_fn(arg)
        if arg_type not in overloaded_types and hasattr(arg_type, '__torch_function__') and (arg_type.__torch_function__ != torch._C._disabled_torch_function_impl):
            if overloaded_types:
                overloaded_types.add(arg_type)
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, get_type_fn(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = {arg_type}
                overloaded_args = [arg]
    return overloaded_args