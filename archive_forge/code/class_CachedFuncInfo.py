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
class CachedFuncInfo:
    """Encapsulates data for a cached function instance.

    CachedFuncInfo instances are scoped to a single script run - they're not
    persistent.
    """

    def __init__(self, func: types.FunctionType, show_spinner: bool | str, allow_widgets: bool, hash_funcs: HashFuncsDict | None):
        self.func = func
        self.show_spinner = show_spinner
        self.allow_widgets = allow_widgets
        self.hash_funcs = hash_funcs

    @property
    def cache_type(self) -> CacheType:
        raise NotImplementedError

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        raise NotImplementedError

    def get_function_cache(self, function_key: str) -> Cache:
        """Get or create the function cache for the given key."""
        raise NotImplementedError