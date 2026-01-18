import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Optional, Type
def _do_enter(self) -> None:
    if self._state != _State.INIT:
        raise RuntimeError(f'invalid state {self._state.value}')
    self._state = _State.ENTER
    self._reschedule()