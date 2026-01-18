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
def add_call_item_to_queue(self):
    while True:
        if self.call_queue.full():
            return
        try:
            work_id = self.work_ids_queue.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = self.pending_work_items[work_id]
            if work_item.future.set_running_or_notify_cancel():
                self.call_queue.put(_CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs), block=True)
            else:
                del self.pending_work_items[work_id]
                continue