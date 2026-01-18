from __future__ import annotations
import contextlib
import functools
import hashlib
import inspect
import math
import os
import pickle
import shutil
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterator, TypeVar, cast, overload
from cachetools import TTLCache
import streamlit as st
from streamlit import config, file_util, util
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.elements.spinner import spinner
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.errors import StreamlitAPIWarning
from streamlit.logger import get_logger
from streamlit.runtime.caching import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
from streamlit.runtime.legacy_caching.hashing import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.util import HASHLIB_KWARGS
class _MemCaches(CacheStatsProvider):
    """Manages all in-memory st.cache caches"""

    def __init__(self):
        self._lock = threading.RLock()
        self._function_caches: dict[str, MemCache] = {}

    def __repr__(self) -> str:
        return util.repr_(self)

    def get_cache(self, key: str, max_entries: float | None, ttl: float | None, display_name: str='') -> MemCache:
        """Return the mem cache for the given key.

        If it doesn't exist, create a new one with the given params.
        """
        if max_entries is None:
            max_entries = math.inf
        if ttl is None:
            ttl = math.inf
        if not isinstance(max_entries, (int, float)):
            raise RuntimeError('max_entries must be an int')
        if not isinstance(ttl, (int, float)):
            raise RuntimeError('ttl must be a float')
        with self._lock:
            mem_cache = self._function_caches.get(key)
            if mem_cache is not None and mem_cache.cache.ttl == ttl and (mem_cache.cache.maxsize == max_entries):
                return mem_cache
            _LOGGER.debug('Creating new mem_cache (key=%s, max_entries=%s, ttl=%s)', key, max_entries, ttl)
            ttl_cache = TTLCache(maxsize=max_entries, ttl=ttl, timer=_TTLCACHE_TIMER)
            mem_cache = MemCache(ttl_cache, display_name)
            self._function_caches[key] = mem_cache
            return mem_cache

    def clear(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        with self._lock:
            function_caches = self._function_caches.copy()
        from streamlit.vendor.pympler.asizeof import asizeof
        stats = [CacheStat('st_cache', cache.display_name, asizeof(c)) for cache in function_caches.values() for c in cache.cache]
        return stats