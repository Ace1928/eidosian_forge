import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def _monitor(self):
    """
        Monitor the queue for records, and ask the handler
        to deal with them.

        This method runs on a separate, internal thread.
        The thread will terminate if it sees a sentinel object in the queue.
        """
    q = self.queue
    has_task_done = hasattr(q, 'task_done')
    while True:
        try:
            record = self.dequeue(True)
            if record is self._sentinel:
                if has_task_done:
                    q.task_done()
                break
            self.handle(record)
            if has_task_done:
                q.task_done()
        except queue.Empty:
            break