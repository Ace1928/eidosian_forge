import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
def _start_thread(self):
    debug('Queue._start_thread()')
    self._buffer.clear()
    self._thread = threading.Thread(target=Queue._feed, args=(self._buffer, self._notempty, self._send_bytes, self._wlock, self._reader.close, self._writer.close, self._ignore_epipe, self._on_queue_feeder_error, self._sem), name='QueueFeederThread')
    self._thread.daemon = True
    debug('doing self._thread.start()')
    self._thread.start()
    debug('... done self._thread.start()')
    if not self._joincancelled:
        self._jointhread = Finalize(self._thread, Queue._finalize_join, [weakref.ref(self._thread)], exitpriority=-5)
    self._close = Finalize(self, Queue._finalize_close, [self._buffer, self._notempty], exitpriority=10)