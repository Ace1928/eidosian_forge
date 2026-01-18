from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def _enqueue_handlers(self, what, *handlers):
    """Enqueues handlers for _run_handlers() to run.

        `what` is the Message being handled, and is used for logging purposes.

        If the background thread with _run_handlers() isn't running yet, starts it.
        """
    with self:
        self._handler_queue.extend(((what, handler) for handler in handlers))
        self._handlers_enqueued.notify_all()
        if len(self._handler_queue) and self._handler_thread is None:
            self._handler_thread = threading.Thread(target=self._run_handlers, name=f'{self} message handler')
            hide_thread_from_debugger(self._handler_thread)
            self._handler_thread.start()