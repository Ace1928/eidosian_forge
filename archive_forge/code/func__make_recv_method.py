import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def _make_recv_method(self, conn):
    get = conn.get
    if hasattr(conn, '_reader'):
        _poll = conn._reader.poll
        if hasattr(conn, 'get_payload') and conn.get_payload:
            get_payload = conn.get_payload

            def _recv(timeout, loads=pickle_loads):
                return (True, loads(get_payload()))
        else:

            def _recv(timeout):
                if _poll(timeout):
                    return (True, get())
                return (False, None)
    else:

        def _recv(timeout):
            try:
                return (True, get(timeout=timeout))
            except Queue.Empty:
                return (False, None)
    return _recv