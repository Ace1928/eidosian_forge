import bmemcached
from oslo_cache._memcache_pool import MemcacheClientPool
from oslo_log import log
class _BMemcacheClient(bmemcached.Client):
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