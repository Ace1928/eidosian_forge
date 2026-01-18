from __future__ import annotations
import math
import threading
from cachetools import TTLCache
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.stats import CacheStat
def _write_to_mem_cache(self, key: str, entry_bytes: bytes) -> None:
    with self._mem_cache_lock:
        self._mem_cache[key] = entry_bytes