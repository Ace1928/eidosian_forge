from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class _Nt(Protocol):
    """Statically typed version of the dtdll.dll functions we use."""

    def RtlNtStatusToDosError(self, status: int, /) -> ErrorCodes:
        ...