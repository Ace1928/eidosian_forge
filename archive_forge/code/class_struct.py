from __future__ import annotations
import logging # isort:skip
from types import SimpleNamespace
from typing import Any, Generic, TypeVar
from .bases import ParameterizedProperty, Property
class struct(SimpleNamespace):
    """
    Allow access unnamed struct with attributes and keys.

    .. note::
        This feature is experimental and may change in the short term.
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, val: Any) -> None:
        setattr(self, key, val)

    def __delitem__(self, key: str) -> None:
        delattr(self, key)