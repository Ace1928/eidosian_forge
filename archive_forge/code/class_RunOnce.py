from functools import _make_key, wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, Type
class RunOnce:
    """Run `func` once, the uniqueness is defined by `key_func`.
    This implementation is serialization safe and thread safe.

    .. note::

        Please use the decorator :func:`~.run_once` instead of directly
        using this class

    :param func: the function to run only once with this wrapper instance
    :param key_func: the unique key determined by arguments of `func`, if not set, it
        will use the same hasing logic as :external+python:func:`functools.lru_cache`
    :param lock_type: lock class type for thread safe
    """

    def __init__(self, func: Callable, key_func: Optional[Callable]=None, lock_type: Type=RLock):
        self._func = func
        if key_func is None:
            self._key_func: Callable = lambda *args, **kwargs: _make_key(args, kwargs, typed=True)
        else:
            self._key_func = key_func
        self._lock_type = lock_type
        self._init_locks()

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d['_lock']
        del d['_locks']
        del d['_store']
        return d

    def __setstate__(self, members: Any) -> None:
        self.__dict__.update(members)
        self._init_locks()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._key_func(*args, **kwargs)
        lock = self._get_lock(key)
        with lock:
            found, res = self._try_get(key)
            if found:
                return res
            res = self._func(*args, **kwargs)
            self._update(key, res)
            return res

    def _get_lock(self, key) -> Any:
        with self._lock:
            if key not in self._locks:
                self._locks[key] = self._lock_type()
            return self._locks[key]

    def _try_get(self, key) -> Tuple[bool, Any]:
        with self._lock:
            if key in self._store:
                return (True, self._store[key])
            return (False, None)

    def _update(self, key, value) -> None:
        with self._lock:
            self._store[key] = value

    def _init_locks(self):
        self._lock = self._lock_type()
        self._locks: Dict[int, Any] = {}
        self._store: Dict[int, Any] = {}