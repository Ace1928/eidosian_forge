import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _is_specialization(cls: type, generic_class: Any) -> bool:
    if generic_class is Annotated and isinstance(cls, _AnnotatedAlias):
        return True
    if not hasattr(cls, '__origin__'):
        return False
    origin = cast(Any, cls).__origin__
    if not inspect.isclass(generic_class):
        generic_class = type(generic_class)
    if not inspect.isclass(origin):
        origin = type(origin)
    return origin is generic_class or issubclass(origin, generic_class)