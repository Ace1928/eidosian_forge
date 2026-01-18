from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
class CachedFunc:

    def __init__(self, info: CachedFuncInfo):
        self._info = info
        self._function_key = _make_function_key(info.cache_type, info.func)

    def __call__(self, *args, **kwargs) -> Any:
        """The wrapper. We'll only call our underlying function on a cache miss."""
        name = self._info.func.__qualname__
        if isinstance(self._info.show_spinner, bool):
            if len(args) == 0 and len(kwargs) == 0:
                message = f'Running `{name}()`.'
            else:
                message = f'Running `{name}(...)`.'
        else:
            message = self._info.show_spinner
        if self._info.show_spinner or isinstance(self._info.show_spinner, str):
            with spinner(message, _cache=True):
                return self._get_or_create_cached_value(args, kwargs)
        else:
            return self._get_or_create_cached_value(args, kwargs)

    def _get_or_create_cached_value(self, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]) -> Any:
        cache = self._info.get_function_cache(self._function_key)
        value_key = _make_value_key(cache_type=self._info.cache_type, func=self._info.func, func_args=func_args, func_kwargs=func_kwargs, hash_funcs=self._info.hash_funcs)
        try:
            cached_result = cache.read_result(value_key)
            return self._handle_cache_hit(cached_result)
        except CacheKeyNotFoundError:
            pass
        return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)

    def _handle_cache_hit(self, result: CachedResult) -> Any:
        """Handle a cache hit: replay the result's cached messages, and return its value."""
        replay_cached_messages(result, self._info.cache_type, self._info.func)
        return result.value

    def _handle_cache_miss(self, cache: Cache, value_key: str, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]) -> Any:
        """Handle a cache miss: compute a new cached value, write it back to the cache,
        and return that newly-computed value.
        """
        with cache.compute_value_lock(value_key):
            try:
                cached_result = cache.read_result(value_key)
                return self._handle_cache_hit(cached_result)
            except CacheKeyNotFoundError:
                pass
            with self._info.cached_message_replay_ctx.calling_cached_function(self._info.func, self._info.allow_widgets):
                computed_value = self._info.func(*func_args, **func_kwargs)
            messages = self._info.cached_message_replay_ctx._most_recent_messages
            try:
                cache.write_result(value_key, computed_value, messages)
                return computed_value
            except (CacheError, RuntimeError):
                if True in [type_util.is_type(computed_value, type_name) for type_name in UNEVALUATED_DATAFRAME_TYPES]:
                    raise UnevaluatedDataFrameError(f'\n                        The function {get_cached_func_name_md(self._info.func)} is decorated with `st.cache_data` but it returns an unevaluated dataframe\n                        of type `{type_util.get_fqn_type(computed_value)}`. Please call `collect()` or `to_pandas()` on the dataframe before returning it,\n                        so `st.cache_data` can serialize and cache it.')
                raise UnserializableReturnValueError(return_value=computed_value, func=self._info.func)

    def clear(self):
        """Clear the wrapped function's associated cache."""
        cache = self._info.get_function_cache(self._function_key)
        cache.clear()