from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
def SetFileCompletionNotificationModes(self, handle: Handle, flags: CompletionModes, /) -> int:
    ...