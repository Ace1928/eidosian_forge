from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class IoControlCodes(enum.IntEnum):
    IOCTL_AFD_POLL = 73764