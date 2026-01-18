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
def _handle_disconnect(self):
    handler = getattr(self.handlers, 'disconnect', lambda: None)
    try:
        handler()
    except Exception:
        log.reraise_exception("Handler {0}\ncouldn't handle disconnect from {1}:", util.srcnameof(handler), self)