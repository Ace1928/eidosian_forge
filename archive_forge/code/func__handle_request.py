import threading
import uuid
import requests.exceptions as exc
from .._compat import queue
def _handle_request(self, kwargs):
    try:
        response = self._session.request(**kwargs)
    except exc.RequestException as e:
        self._exceptions.put((kwargs, e))
    else:
        self._responses.put((kwargs, response))
    finally:
        self._jobs.task_done()