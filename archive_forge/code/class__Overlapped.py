from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class _Overlapped(Protocol):
    Internal: int
    InternalHigh: int
    DUMMYUNIONNAME: _DummyUnion
    hEvent: Handle