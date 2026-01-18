from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
def CreateIoCompletionPort(self, FileHandle: Handle, ExistingCompletionPort: CData | AlwaysNull, CompletionKey: int, NumberOfConcurrentThreads: int, /) -> Handle:
    ...