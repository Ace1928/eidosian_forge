from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, cast, overload
from datetime import date, datetime
from typing_extensions import Self
import pydantic
from pydantic.fields import FieldInfo
from ._types import StrBytesIntFloat
class typed_cached_property(Generic[_T]):
    func: Callable[[Any], _T]
    attrname: str | None

    def __init__(self, func: Callable[[Any], _T]) -> None:
        ...

    @overload
    def __get__(self, instance: None, owner: type[Any] | None=None) -> Self:
        ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None=None) -> _T:
        ...

    def __get__(self, instance: object, owner: type[Any] | None=None) -> _T | Self:
        raise NotImplementedError()

    def __set_name__(self, owner: type[Any], name: str) -> None:
        ...

    def __set__(self, instance: object, value: _T) -> None:
        ...