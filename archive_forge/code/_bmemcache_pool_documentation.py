import bmemcached
from oslo_cache._memcache_pool import MemcacheClientPool
from oslo_log import log
Thread global memcache client

    As client is inherited from threading.local we have to restore object
    methods overloaded by threading.local so we can reuse clients in
    different threads
    