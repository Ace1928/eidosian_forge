from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
def _set_contents(self, c: list[tuple[Widget, typing.Any]]) -> None:
    warnings.warn(f'method `{self.__class__.__name__}._set_contents` is deprecated, please use `{self.__class__.__name__}.contents` property', DeprecationWarning, stacklevel=2)
    self.contents = c