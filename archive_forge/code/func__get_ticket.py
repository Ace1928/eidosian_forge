import queue
import threading
import time
def _get_ticket(self):
    ticket = threading.Event()
    with self._lock:
        if self._count <= self._concurrency:
            self._count += 1
            return self._advance_timer()
        else:
            self._queue.put(ticket)
    ticket.wait()
    with self._lock:
        return self._advance_timer()