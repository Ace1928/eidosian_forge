import atexit
from concurrent.futures import _base
import Queue as queue
import multiprocessing
import threading
import weakref
import sys
def _start_queue_management_thread(self):

    def weakref_cb(_, q=self._result_queue):
        q.put(None)
    if self._queue_management_thread is None:
        self._queue_management_thread = threading.Thread(target=_queue_management_worker, args=(weakref.ref(self, weakref_cb), self._processes, self._pending_work_items, self._work_ids, self._call_queue, self._result_queue))
        self._queue_management_thread.daemon = True
        self._queue_management_thread.start()
        _threads_queues[self._queue_management_thread] = self._result_queue