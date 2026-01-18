from __future__ import annotations
import math
import os
import shutil
from typing import Final
from streamlit import util
from streamlit.file_util import get_streamlit_file_path, streamlit_read, streamlit_write
from streamlit.logger import get_logger
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
def _is_cache_file(self, fname: str) -> bool:
    """Return true if the given file name is a cache file for this storage."""
    return fname.startswith(f'{self.function_key}-') and fname.endswith(f'.{_CACHED_FILE_EXTENSION}')