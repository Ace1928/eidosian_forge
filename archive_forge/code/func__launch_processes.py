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
def _launch_processes(self):
    assert not self._executor_manager_thread, 'Processes cannot be fork()ed after the thread has started, deadlock in the child processes could result.'
    for _ in range(len(self._processes), self._max_workers):
        self._spawn_process()