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
def _start_executor_manager_thread(self):
    if self._executor_manager_thread is None:
        if not self._safe_to_dynamically_spawn_children:
            self._launch_processes()
        self._executor_manager_thread = _ExecutorManagerThread(self)
        self._executor_manager_thread.start()
        _threads_wakeups[self._executor_manager_thread] = self._executor_manager_thread_wakeup