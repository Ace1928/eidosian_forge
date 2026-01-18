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
class _InternalState:
    """
    An L{_InternalState} is a helper object for a L{_ModuleProxy}, so that it
    can easily access its own attributes, bypassing its logic for delegating to
    another object that it's proxying for.

    @ivar proxy: a L{_ModuleProxy}
    """

    def __init__(self, proxy):
        object.__setattr__(self, 'proxy', proxy)

    def __getattribute__(self, name):
        return object.__getattribute__(object.__getattribute__(self, 'proxy'), name)

    def __setattr__(self, name, value):
        return object.__setattr__(object.__getattribute__(self, 'proxy'), name, value)