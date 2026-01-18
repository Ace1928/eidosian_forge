from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload
@classmethod
def _ensure_method(cls, fn: _GetterCallable[_T] | _GetterClassMethod[_T] | _SetterCallable[_T] | _SetterClassMethod[_T]) -> _GetterClassMethod[_T] | _SetterClassMethod[_T]:
    """
        Ensure fn is a classmethod or staticmethod.
        """
    needs_method = not isinstance(fn, (classmethod, staticmethod))
    return classmethod(fn) if needs_method else fn