import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
class AttrCleaner(ContextManager[None]):
    """A context manager that tries to make an object not exhibit side-effects
    on attribute lookup.

    Unless explicitly required, prefer `getattr_safe`."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __enter__(self) -> None:
        """Try to make an object not exhibit side-effects on attribute
        lookup."""
        type_ = type(self._obj)
        __getattr__ = getattr(type_, '__getattr__', None)
        if __getattr__ is not None:
            try:
                setattr(type_, '__getattr__', lambda *_, **__: None)
            except TypeError:
                __getattr__ = None
        __getattribute__ = getattr(type_, '__getattribute__', None)
        if __getattribute__ is not None:
            try:
                setattr(type_, '__getattribute__', object.__getattribute__)
            except TypeError:
                __getattribute__ = None
        self._attribs = (__getattribute__, __getattr__)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        """Restore an object's magic methods."""
        type_ = type(self._obj)
        __getattribute__, __getattr__ = self._attribs
        if __getattribute__ is not None:
            setattr(type_, '__getattribute__', __getattribute__)
        if __getattr__ is not None:
            setattr(type_, '__getattr__', __getattr__)
        return False