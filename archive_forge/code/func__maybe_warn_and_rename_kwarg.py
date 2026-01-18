from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def _maybe_warn_and_rename_kwarg(args: tuple[Any, ...], kwargs: dict[str, Any], *, func_name: str, original_func_co_varnames: tuple[str, ...], old_arg_name: str, new_alias: str | None, warning_msg: str, category: Type[Warning], predicate: Callable[[Any], bool] | None) -> None:
    arg_names_to_values = {name: val for val, name in zip(args, original_func_co_varnames)}
    arg_names_to_values.update(kwargs)
    if old_arg_name not in arg_names_to_values:
        return
    if new_alias and new_alias in arg_names_to_values:
        raise TypeError(f'{func_name} received both {new_alias} and {old_arg_name} (deprecated).')
    val = arg_names_to_values[old_arg_name]
    if predicate and (not predicate(val)):
        return
    warnings.warn(warning_msg, category=category, stacklevel=3)
    if new_alias is not None:
        kwargs[new_alias] = val