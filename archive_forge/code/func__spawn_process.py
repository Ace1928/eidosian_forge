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
def _spawn_process(self):
    p = self._mp_context.Process(target=_process_worker, args=(self._call_queue, self._result_queue, self._initializer, self._initargs, self._max_tasks_per_child))
    p.start()
    self._processes[p.pid] = p