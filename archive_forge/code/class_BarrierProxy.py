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
class BarrierProxy(BaseProxy):
    _exposed_ = ('__getattribute__', 'wait', 'abort', 'reset')

    def wait(self, timeout=None):
        return self._callmethod('wait', (timeout,))

    def abort(self):
        return self._callmethod('abort')

    def reset(self):
        return self._callmethod('reset')

    @property
    def parties(self):
        return self._callmethod('__getattribute__', ('parties',))

    @property
    def n_waiting(self):
        return self._callmethod('__getattribute__', ('n_waiting',))

    @property
    def broken(self):
        return self._callmethod('__getattribute__', ('broken',))