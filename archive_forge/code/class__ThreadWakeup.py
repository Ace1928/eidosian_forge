import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
class _ThreadWakeup:

    def __init__(self):
        self._closed = False
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        if not self._closed:
            self._closed = True
            self._writer.close()
            self._reader.close()

    def wakeup(self):
        if not self._closed:
            self._writer.send_bytes(b'')

    def clear(self):
        if not self._closed:
            while self._reader.poll():
                self._reader.recv_bytes()