import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
class EventProxy(BaseProxy):
    _exposed_ = ('is_set', 'set', 'clear', 'wait')

    def is_set(self):
        return self._callmethod('is_set')

    def set(self):
        return self._callmethod('set')

    def clear(self):
        return self._callmethod('clear')

    def wait(self, timeout=None):
        return self._callmethod('wait', (timeout,))