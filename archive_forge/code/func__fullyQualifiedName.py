from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def _fullyQualifiedName(obj):
    """
    Return the fully qualified name of a module, class, method or function.
    Classes and functions need to be module level ones to be correctly
    qualified.

    @rtype: C{str}.
    """
    try:
        name = obj.__qualname__
    except AttributeError:
        name = obj.__name__
    if inspect.isclass(obj) or inspect.isfunction(obj):
        moduleName = obj.__module__
        return f'{moduleName}.{name}'
    elif inspect.ismethod(obj):
        return f'{obj.__module__}.{obj.__qualname__}'
    return name