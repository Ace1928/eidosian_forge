import contextvars
import enum
import functools
import threading
import signal
import sys
from . import coroutines
from . import events
from . import exceptions
from . import tasks
def _on_sigint(self, signum, frame, main_task):
    self._interrupt_count += 1
    if self._interrupt_count == 1 and (not main_task.done()):
        main_task.cancel()
        self._loop.call_soon_threadsafe(lambda: None)
        return
    raise KeyboardInterrupt()