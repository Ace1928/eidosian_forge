import os
def _thread_worker_timeout(self, seconds, msg, timeout):
    with eventlet.Timeout(timeout):
        try:
            self._thread_worker(seconds, msg)
        except eventlet.Timeout:
            pass