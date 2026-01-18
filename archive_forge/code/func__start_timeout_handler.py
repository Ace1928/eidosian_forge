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
def _start_timeout_handler(self):
    if self.threads and self._timeout_handler is not None:
        with self._timeout_handler_mutex:
            if not self._timeout_handler_started:
                self._timeout_handler_started = True
                self._timeout_handler.start()