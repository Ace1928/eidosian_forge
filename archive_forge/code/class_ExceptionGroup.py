from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload
class ExceptionGroup(BaseExceptionGroup[_ExceptionT_co], Exception):

    def __new__(cls, __message: str, __exceptions: Sequence[_ExceptionT_co]) -> Self:
        return super().__new__(cls, __message, __exceptions)
    if TYPE_CHECKING:

        @property
        def exceptions(self) -> tuple[_ExceptionT_co | ExceptionGroup[_ExceptionT_co], ...]:
            ...

        @overload
        def subgroup(self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]) -> ExceptionGroup[_ExceptionT] | None:
            ...

        @overload
        def subgroup(self, __condition: Callable[[_ExceptionT_co | Self], bool]) -> ExceptionGroup[_ExceptionT_co] | None:
            ...

        def subgroup(self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...] | Callable[[_ExceptionT_co], bool]) -> ExceptionGroup[_ExceptionT] | None:
            return super().subgroup(__condition)

        @overload
        def split(self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]) -> tuple[ExceptionGroup[_ExceptionT] | None, ExceptionGroup[_ExceptionT_co] | None]:
            ...

        @overload
        def split(self, __condition: Callable[[_ExceptionT_co | Self], bool]) -> tuple[ExceptionGroup[_ExceptionT_co] | None, ExceptionGroup[_ExceptionT_co] | None]:
            ...

        def split(self: Self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...] | Callable[[_ExceptionT_co], bool]) -> tuple[ExceptionGroup[_ExceptionT_co] | None, ExceptionGroup[_ExceptionT_co] | None]:
            return super().split(__condition)