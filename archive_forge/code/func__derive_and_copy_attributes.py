from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload
def _derive_and_copy_attributes(self, excs):
    eg = self.derive(excs)
    eg.__cause__ = self.__cause__
    eg.__context__ = self.__context__
    eg.__traceback__ = self.__traceback__
    if hasattr(self, '__notes__'):
        eg.__notes__ = list(self.__notes__)
    return eg