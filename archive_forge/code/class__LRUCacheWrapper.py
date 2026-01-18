import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
@final
class _LRUCacheWrapper(Generic[_R]):

    def __init__(self, fn: _CB[_R], maxsize: Optional[int], typed: bool, ttl: Optional[float]) -> None:
        try:
            self.__module__ = fn.__module__
        except AttributeError:
            pass
        try:
            self.__name__ = fn.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = fn.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = fn.__doc__
        except AttributeError:
            pass
        try:
            self.__annotations__ = fn.__annotations__
        except AttributeError:
            pass
        try:
            self.__dict__.update(fn.__dict__)
        except AttributeError:
            pass
        self._is_coroutine = _is_coroutine
        self.__wrapped__ = fn
        self.__maxsize = maxsize
        self.__typed = typed
        self.__ttl = ttl
        self.__cache: OrderedDict[Hashable, _CacheItem[_R]] = OrderedDict()
        self.__closed = False
        self.__hits = 0
        self.__misses = 0
        self.__tasks: Set['asyncio.Task[_R]'] = set()

    def cache_invalidate(self, /, *args: Hashable, **kwargs: Any) -> bool:
        key = _make_key(args, kwargs, self.__typed)
        cache_item = self.__cache.pop(key, None)
        if cache_item is None:
            return False
        else:
            cache_item.cancel()
            return True

    def cache_clear(self) -> None:
        self.__hits = 0
        self.__misses = 0
        for c in self.__cache.values():
            if c.later_call:
                c.later_call.cancel()
        self.__cache.clear()
        self.__tasks.clear()

    async def cache_close(self, *, wait: bool=False) -> None:
        self.__closed = True
        tasks = list(self.__tasks)
        if not tasks:
            return
        if not wait:
            for task in tasks:
                if not task.done():
                    task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def cache_info(self) -> _CacheInfo:
        return _CacheInfo(self.__hits, self.__misses, self.__maxsize, len(self.__cache))

    def cache_parameters(self) -> _CacheParameters:
        return _CacheParameters(maxsize=self.__maxsize, typed=self.__typed, tasks=len(self.__tasks), closed=self.__closed)

    def _cache_hit(self, key: Hashable) -> None:
        self.__hits += 1
        self.__cache.move_to_end(key)

    def _cache_miss(self, key: Hashable) -> None:
        self.__misses += 1

    def _task_done_callback(self, fut: 'asyncio.Future[_R]', key: Hashable, task: 'asyncio.Task[_R]') -> None:
        self.__tasks.discard(task)
        cache_item = self.__cache.get(key)
        if self.__ttl is not None and cache_item is not None:
            loop = asyncio.get_running_loop()
            cache_item.later_call = loop.call_later(self.__ttl, self.__cache.pop, key, None)
        if task.cancelled():
            fut.cancel()
            return
        exc = task.exception()
        if exc is not None:
            fut.set_exception(exc)
            return
        fut.set_result(task.result())

    async def __call__(self, /, *fn_args: Any, **fn_kwargs: Any) -> _R:
        if self.__closed:
            raise RuntimeError(f'alru_cache is closed for {self}')
        loop = asyncio.get_running_loop()
        key = _make_key(fn_args, fn_kwargs, self.__typed)
        cache_item = self.__cache.get(key)
        if cache_item is not None:
            if not cache_item.fut.done():
                self._cache_hit(key)
                return await asyncio.shield(cache_item.fut)
            exc = cache_item.fut._exception
            if exc is None:
                self._cache_hit(key)
                return cache_item.fut.result()
            else:
                cache_item = self.__cache.pop(key)
                cache_item.cancel()
        fut = loop.create_future()
        coro = self.__wrapped__(*fn_args, **fn_kwargs)
        task: asyncio.Task[_R] = loop.create_task(coro)
        self.__tasks.add(task)
        task.add_done_callback(partial(self._task_done_callback, fut, key))
        self.__cache[key] = _CacheItem(fut, None)
        if self.__maxsize is not None and len(self.__cache) > self.__maxsize:
            dropped_key, cache_item = self.__cache.popitem(last=False)
            cache_item.cancel()
        self._cache_miss(key)
        return await asyncio.shield(fut)

    def __get__(self, instance: _T, owner: Optional[Type[_T]]) -> Union[Self, '_LRUCacheWrapperInstanceMethod[_R, _T]']:
        if owner is None:
            return self
        else:
            return _LRUCacheWrapperInstanceMethod(self, instance)