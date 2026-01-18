import inspect
import os
from argparse import Namespace
from ast import literal_eval
from contextlib import suppress
from functools import wraps
from typing import Any, Callable, Type, TypeVar, cast
def _defaults_from_env_vars(fn: _T) -> _T:

    @wraps(fn)
    def insert_env_defaults(self: Any, *args: Any, **kwargs: Any) -> Any:
        cls = self.__class__
        if args:
            cls_arg_names = inspect.signature(cls).parameters
            kwargs.update(dict(zip(cls_arg_names, args)))
        env_variables = vars(_parse_env_variables(cls))
        kwargs = dict(list(env_variables.items()) + list(kwargs.items()))
        return fn(self, **kwargs)
    return cast(_T, insert_env_defaults)