import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
class classproperty_v2(Generic[T, R]):

    def __init__(self, func: Callable[[Type[T]], R]) -> None:
        self.func = func

    def __get__(self, obj: Any, cls: Type[T]) -> R:
        return self.func(cls)