from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
class DataLoader(local):
    batch = True
    max_batch_size = None
    cache = True

    def __init__(self, batch_load_fn=None, batch=None, max_batch_size=None, cache=None, get_cache_key=None, cache_map=None, scheduler=None):
        if batch_load_fn is not None:
            self.batch_load_fn = batch_load_fn
        if not callable(self.batch_load_fn):
            raise TypeError('DataLoader must be have a batch_load_fn which accepts List<key> and returns Promise<List<value>>, but got: {}.'.format(batch_load_fn))
        if batch is not None:
            self.batch = batch
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if cache is not None:
            self.cache = cache
        self.get_cache_key = get_cache_key or (lambda x: x)
        self._promise_cache = cache_map or {}
        self._queue = []
        self._scheduler = scheduler

    def load(self, key=None):
        """
        Loads a key, returning a `Promise` for the value represented by that key.
        """
        if key is None:
            raise TypeError(('The loader.load() function must be called with a value,' + 'but got: {}.').format(key))
        cache_key = self.get_cache_key(key)
        if self.cache:
            cached_promise = self._promise_cache.get(cache_key)
            if cached_promise:
                return cached_promise
        promise = Promise(partial(self.do_resolve_reject, key))
        if self.cache:
            self._promise_cache[cache_key] = promise
        return promise

    def do_resolve_reject(self, key, resolve, reject):
        self._queue.append(Loader(key=key, resolve=resolve, reject=reject))
        if len(self._queue) == 1:
            if self.batch:
                enqueue_post_promise_job(partial(dispatch_queue, self), self._scheduler)
            else:
                dispatch_queue(self)

    def load_many(self, keys):
        """
        Loads multiple keys, promising an array of values

        >>> a, b = await my_loader.load_many([ 'a', 'b' ])

        This is equivalent to the more verbose:

        >>> a, b = await Promise.all([
        >>>    my_loader.load('a'),
        >>>    my_loader.load('b')
        >>> ])
        """
        if not isinstance(keys, Iterable):
            raise TypeError(('The loader.loadMany() function must be called with Array<key> ' + 'but got: {}.').format(keys))
        return Promise.all([self.load(key) for key in keys])

    def clear(self, key):
        """
        Clears the value at `key` from the cache, if it exists. Returns itself for
        method chaining.
        """
        cache_key = self.get_cache_key(key)
        self._promise_cache.pop(cache_key, None)
        return self

    def clear_all(self):
        """
        Clears the entire cache. To be used when some event results in unknown
        invalidations across this particular `DataLoader`. Returns itself for
        method chaining.
        """
        self._promise_cache.clear()
        return self

    def prime(self, key, value):
        """
        Adds the provied key and value to the cache. If the key already exists, no
        change is made. Returns itself for method chaining.
        """
        cache_key = self.get_cache_key(key)
        if cache_key not in self._promise_cache:
            if isinstance(value, Exception):
                promise = Promise.reject(value)
            else:
                promise = Promise.resolve(value)
            self._promise_cache[cache_key] = promise
        return self