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
def _enqueue_response_handlers(self):
    response = self.response
    if response is None:
        return

    def run_handlers():
        for handler in handlers:
            try:
                try:
                    handler(response)
                except MessageHandlingError as exc:
                    if not exc.applies_to(response):
                        raise
                    log.error("Handler {0}\ncouldn't handle {1}:\n{2}", util.srcnameof(handler), response.describe(), str(exc))
            except Exception:
                log.reraise_exception("Handler {0}\ncouldn't handle {1}:", util.srcnameof(handler), response.describe())
    handlers = self._response_handlers[:]
    self.channel._enqueue_handlers(response, run_handlers)
    del self._response_handlers[:]