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
class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(self, max_size=0, *, ctx, pending_work_items, shutdown_lock, thread_wakeup):
        self.pending_work_items = pending_work_items
        self.shutdown_lock = shutdown_lock
        self.thread_wakeup = thread_wakeup
        super().__init__(max_size, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            tb = format_exception(type(e), e, e.__traceback__)
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format(''.join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            with self.shutdown_lock:
                self.thread_wakeup.wakeup()
            if work_item is not None:
                work_item.future.set_exception(e)
        else:
            super()._on_queue_feeder_error(e, obj)