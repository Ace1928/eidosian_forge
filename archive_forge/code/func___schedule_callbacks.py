import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def __schedule_callbacks(self):
    """Internal: Ask the event loop to call all callbacks.

        The callbacks are scheduled to be called as soon as possible. Also
        clears the callback list.
        """
    callbacks = self._callbacks[:]
    if not callbacks:
        return
    self._callbacks[:] = []
    for callback, ctx in callbacks:
        self._loop.call_soon(callback, self, context=ctx)