from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo):
    sentinel = object()
    make_key = _make_key
    PREV, NEXT, KEY, RESULT = (0, 1, 2, 3)
    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get
    cache_len = cache.__len__
    lock = RLock()
    root = []
    root[:] = [root, root, None, None]
    if maxsize == 0:

        def wrapper(*args, **kwds):
            nonlocal misses
            misses += 1
            result = user_function(*args, **kwds)
            return result
    elif maxsize is None:

        def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result
    else:

        def wrapper(*args, **kwds):
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            with lock:
                if key in cache:
                    pass
                elif full:
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    oldresult = root[RESULT]
                    root[KEY] = root[RESULT] = None
                    del cache[oldkey]
                    cache[key] = oldroot
                else:
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    full = cache_len() >= maxsize
            return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False
    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper