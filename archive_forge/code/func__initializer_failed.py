from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os
def _initializer_failed(self):
    with self._shutdown_lock:
        self._broken = 'A thread initializer failed, the thread pool is not usable anymore'
        while True:
            try:
                work_item = self._work_queue.get_nowait()
            except queue.Empty:
                break
            if work_item is not None:
                work_item.future.set_exception(BrokenThreadPool(self._broken))