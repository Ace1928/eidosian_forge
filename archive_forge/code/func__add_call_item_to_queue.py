import atexit
from concurrent.futures import _base
import Queue as queue
import multiprocessing
import threading
import weakref
import sys
def _add_call_item_to_queue(pending_work_items, work_ids, call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]
            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(_CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs), block=True)
            else:
                del pending_work_items[work_id]
                continue