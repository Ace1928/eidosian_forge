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
def _adjust_process_count(self):
    if self._idle_worker_semaphore.acquire(blocking=False):
        return
    process_count = len(self._processes)
    if process_count < self._max_workers:
        self._spawn_process()