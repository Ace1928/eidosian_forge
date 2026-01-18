from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
def PostQueuedCompletionStatus(self, CompletionPort: Handle, dwNumberOfBytesTransferred: int, dwCompletionKey: int, lpOverlapped: CData | AlwaysNull, /) -> bool:
    ...