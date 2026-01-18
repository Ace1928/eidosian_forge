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
def _run_handlers(self):
    """Runs enqueued handlers until the channel is closed, or until the handler
        queue is empty once the channel is closed.
        """
    while True:
        with self:
            closed = self._closed
        if closed:
            self._parser_thread.join()
        with self:
            if not closed and (not len(self._handler_queue)):
                self._handlers_enqueued.wait()
            handlers = self._handler_queue[:]
            del self._handler_queue[:]
            if closed and (not len(handlers)):
                self._handler_thread = None
                return
        for what, handler in handlers:
            if closed and handler in (Event._handle, Request._handle):
                continue
            with log.prefixed('/handling {0}/\n', what.describe()):
                try:
                    handler()
                except Exception:
                    self.close()
                    os._exit(1)