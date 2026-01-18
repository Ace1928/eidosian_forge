from __future__ import annotations as _annotations
import types
import typing
from typing import Any
import typing_extensions
from . import _typing_extra
def __pretty__(self, fmt: typing.Callable[[Any], Any], **kwargs: Any) -> typing.Generator[Any, None, None]:
    """Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects."""
    yield (self.__repr_name__() + '(')
    yield 1
    for name, value in self.__repr_args__():
        if name is not None:
            yield (name + '=')
        yield fmt(value)
        yield ','
        yield 0
    yield (-1)
    yield ')'