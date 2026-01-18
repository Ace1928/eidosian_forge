import asyncio
import warnings
from types import TracebackType
from typing import Any  # noqa
from typing import Optional, Type
def _cancel_task(self) -> None:
    if self._task is not None:
        self._task.cancel()
        self._cancelled = True