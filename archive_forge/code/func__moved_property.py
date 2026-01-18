import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
def _moved_property(new_name: str, old_name: ty.Optional[str]=None, target: ty.Optional[str]=None) -> ty.Any:

    def getter(self: ty.Any) -> ty.Any:
        _moved_msg(new_name, old_name)
        return getattr(self, target or new_name)

    def setter(self: ty.Any, value: str) -> None:
        _moved_msg(new_name, old_name)
        setattr(self, target or new_name, value)

    def deleter(self: ty.Any) -> None:
        _moved_msg(new_name, old_name)
        delattr(self, target or new_name)
    return property(getter, setter, deleter)