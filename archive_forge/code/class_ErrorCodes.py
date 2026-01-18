from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class ErrorCodes(enum.IntEnum):
    STATUS_TIMEOUT = 258
    WAIT_TIMEOUT = 258
    WAIT_ABANDONED = 128
    WAIT_OBJECT_0 = 0
    WAIT_FAILED = 4294967295
    ERROR_IO_PENDING = 997
    ERROR_OPERATION_ABORTED = 995
    ERROR_ABANDONED_WAIT_0 = 735
    ERROR_INVALID_HANDLE = 6
    ERROR_INVALID_PARMETER = 87
    ERROR_NOT_FOUND = 1168
    ERROR_NOT_SOCKET = 10038