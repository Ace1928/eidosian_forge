import enum
from types import TracebackType
from typing import final, Optional, Type
from . import events
from . import exceptions
from . import tasks
def _on_timeout(self) -> None:
    assert self._state is _State.ENTERED
    self._task.cancel()
    self._state = _State.EXPIRING
    self._timeout_handler = None