import multiprocessing
import requests
from . import thread
from .._compat import queue
class ThreadProxy(object):
    proxied_attr = None

    def __getattr__(self, attr):
        """Proxy attribute accesses to the proxied object."""
        get = object.__getattribute__
        if attr not in self.attrs:
            response = get(self, self.proxied_attr)
            return getattr(response, attr)
        else:
            return get(self, attr)