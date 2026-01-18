from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
def GetQueuedCompletionStatusEx(self, CompletionPort: Handle, lpCompletionPortEntries: CData, ulCount: int, ulNumEntriesRemoved: CData, dwMilliseconds: int, fAlertable: bool | int, /) -> CData:
    ...