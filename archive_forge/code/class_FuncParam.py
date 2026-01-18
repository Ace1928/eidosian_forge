import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class FuncParam:
    """Function paramter. It defers the function call after all its parameters
    are no longer tuning parameters

    :param func: function to generate parameter value
    :param args: list arguments
    :param kwargs: key-value arguments

    .. code-block:: python

        s = Space(a=1, b=FuncParam(lambda x, y: x + y, x=Grid(0, 1), y=Grid(3, 4)))
        assert [
            dict(a=1, b=3),
            dict(a=1, b=4),
            dict(a=1, b=4),
            dict(a=1, b=5),
        ] == list(s)
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any):
        self._func = func
        self._args = list(args)
        self._kwargs = dict(kwargs)

    def __uuid__(self) -> str:
        """Unique id for this expression"""
        return to_uuid(get_full_type_path(self._func), self._args, self._kwargs)

    def __call__(self) -> Any:
        """Call the function to generate value"""
        return self._func(*self._args, **self._kwargs)

    def __setitem__(self, key: Any, item: Any) -> None:
        """Update argument value

        :param key: key to set, if int, then set in ``args`` else set in ``kwargs``
        :param item: value to use
        """
        if isinstance(key, int):
            self._args[key] = item
        else:
            self._kwargs[key] = item

    def __getitem__(self, key: Any) -> Any:
        """Get argument value

        :param key: key to get, if int, then get in ``args`` else get in ``kwargs``
        :return: the correspondent value
        """
        if isinstance(key, int):
            return self._args[key]
        else:
            return self._kwargs[key]

    def __eq__(self, other: Any) -> bool:
        """Whether the expression equals to the other one

        :param other: another ``FuncParam``
        :return: whether they are equal
        """
        return self._func is other._func and self._args == other._args and (self._kwargs == other._kwargs)

    def __repr__(self) -> str:
        a: List[str] = [self._func.__name__]
        a += [repr(x) for x in self._args]
        a += [f'{k}={repr(v)}' for k, v in self._kwargs.items()]
        return 'FuncParam(' + ', '.join(a) + ')'