import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Optional, Type
def _do_exit(self, exc_type: Optional[Type[BaseException]]) -> None:
    if exc_type is asyncio.CancelledError and self._state == _State.TIMEOUT:
        self._timeout_handler = None
        raise asyncio.TimeoutError
    self._state = _State.EXIT
    self._reject()
    return None