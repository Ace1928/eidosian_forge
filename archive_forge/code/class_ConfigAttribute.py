from __future__ import annotations
import errno
import json
import os
import types
import typing as t
from werkzeug.utils import import_string
class ConfigAttribute(t.Generic[T]):
    """Makes an attribute forward to the config"""

    def __init__(self, name: str, get_converter: t.Callable[[t.Any], T] | None=None) -> None:
        self.__name__ = name
        self.get_converter = get_converter

    @t.overload
    def __get__(self, obj: None, owner: None) -> te.Self:
        ...

    @t.overload
    def __get__(self, obj: App, owner: type[App]) -> T:
        ...

    def __get__(self, obj: App | None, owner: type[App] | None=None) -> T | te.Self:
        if obj is None:
            return self
        rv = obj.config[self.__name__]
        if self.get_converter is not None:
            rv = self.get_converter(rv)
        return rv

    def __set__(self, obj: App, value: t.Any) -> None:
        obj.config[self.__name__] = value