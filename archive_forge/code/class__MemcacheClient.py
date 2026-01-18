import collections
import contextlib
import itertools
import queue
import threading
import time
import memcache
from oslo_log import log
from oslo_cache._i18n import _
from oslo_cache import exception
class _MemcacheClient(memcache.Client):
    """Thread global memcache client

    As client is inherited from threading.local we have to restore object
    methods overloaded by threading.local so we can reuse clients in
    different threads
    """
    __delattr__ = object.__delattr__
    __getattribute__ = object.__getattribute__
    __setattr__ = object.__setattr__
    if eventlet and eventlet.patcher.is_monkey_patched('thread'):

        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)
    else:
        __new__ = object.__new__

    def __del__(self):
        pass