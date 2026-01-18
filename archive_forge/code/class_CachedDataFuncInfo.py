from __future__ import annotations
import pickle
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, Literal, TypeVar, Union, cast, overload
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import CacheError, CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.caching.storage import (
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.dummy_cache_storage import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
class CachedDataFuncInfo(CachedFuncInfo):
    """Implements the CachedFuncInfo interface for @st.cache_data"""

    def __init__(self, func: types.FunctionType, show_spinner: bool | str, persist: CachePersistType, max_entries: int | None, ttl: float | timedelta | str | None, allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        super().__init__(func, show_spinner=show_spinner, allow_widgets=allow_widgets, hash_funcs=hash_funcs)
        self.persist = persist
        self.max_entries = max_entries
        self.ttl = ttl
        self.validate_params()

    @property
    def cache_type(self) -> CacheType:
        return CacheType.DATA

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        return CACHE_DATA_MESSAGE_REPLAY_CTX

    @property
    def display_name(self) -> str:
        """A human-readable name for the cached function"""
        return f'{self.func.__module__}.{self.func.__qualname__}'

    def get_function_cache(self, function_key: str) -> Cache:
        return _data_caches.get_cache(key=function_key, persist=self.persist, max_entries=self.max_entries, ttl=self.ttl, display_name=self.display_name, allow_widgets=self.allow_widgets)

    def validate_params(self) -> None:
        """
        Validate the params passed to @st.cache_data are compatible with cache storage

        When called, this method could log warnings if cache params are invalid
        for current storage.
        """
        _data_caches.validate_cache_params(function_name=self.func.__name__, persist=self.persist, max_entries=self.max_entries, ttl=self.ttl)