import threading
import uuid
import requests.exceptions as exc
from .._compat import queue
def _create_worker(self):
    self._worker = threading.Thread(target=self._make_request, name=uuid.uuid4())
    self._worker.daemon = True
    self._worker._state = 0
    self._worker.start()