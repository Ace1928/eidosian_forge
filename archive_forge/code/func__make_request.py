import threading
import uuid
import requests.exceptions as exc
from .._compat import queue
def _make_request(self):
    while True:
        try:
            kwargs = self._jobs.get_nowait()
        except queue.Empty:
            break
        self._handle_request(kwargs)