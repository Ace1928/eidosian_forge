import multiprocessing
import requests
from . import thread
from .._compat import queue
class ThreadResponse(ThreadProxy):
    """A wrapper around a requests Response object.

    This will proxy most attribute access actions to the Response object. For
    example, if you wanted the parsed JSON from the response, you might do:

    .. code-block:: python

        thread_response = pool.get_response()
        json = thread_response.json()

    """
    proxied_attr = 'response'
    attrs = frozenset(['request_kwargs', 'response'])

    def __init__(self, request_kwargs, response):
        self.request_kwargs = request_kwargs
        self.response = response