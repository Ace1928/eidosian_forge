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
class LocalDiskCacheStorage(CacheStorage):
    """Cache storage that persists data to disk
    This is the default cache persistence layer for `@st.cache_data`
    """

    def __init__(self, context: CacheStorageContext):
        self.function_key = context.function_key
        self.persist = context.persist
        self._ttl_seconds = context.ttl_seconds
        self._max_entries = context.max_entries

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds if self._ttl_seconds is not None else math.inf

    @property
    def max_entries(self) -> float:
        return float(self._max_entries) if self._max_entries is not None else math.inf

    def get(self, key: str) -> bytes:
        """
        Returns the stored value for the key if persisted,
        raise CacheStorageKeyNotFoundError if not found, or not configured
        with persist="disk"
        """
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                with streamlit_read(path, binary=True) as input:
                    value = input.read()
                    _LOGGER.debug('Disk cache HIT: %s', key)
                    return bytes(value)
            except FileNotFoundError:
                raise CacheStorageKeyNotFoundError('Key not found in disk cache')
            except Exception as ex:
                _LOGGER.error(ex)
                raise CacheStorageError('Unable to read from cache') from ex
        else:
            raise CacheStorageKeyNotFoundError(f'Local disk cache storage is disabled (persist={self.persist})')

    def set(self, key: str, value: bytes) -> None:
        """Sets the value for a given key"""
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                with streamlit_write(path, binary=True) as output:
                    output.write(value)
            except util.Error as e:
                _LOGGER.debug(e)
                try:
                    os.remove(path)
                except (FileNotFoundError, OSError):
                    pass
                raise CacheStorageError('Unable to write to cache') from e

    def delete(self, key: str) -> None:
        """Delete a cache file from disk. If the file does not exist on disk,
        return silently. If another exception occurs, log it. Does not throw.
        """
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as ex:
                _LOGGER.exception('Unable to remove a file from the disk cache', exc_info=ex)

    def clear(self) -> None:
        """Delete all keys for the current storage"""
        cache_dir = get_cache_folder_path()
        if os.path.isdir(cache_dir):
            for file_name in os.listdir(cache_dir):
                if self._is_cache_file(file_name):
                    os.remove(os.path.join(cache_dir, file_name))

    def close(self) -> None:
        """Dummy implementation of close, we don't need to actually "close" anything"""

    def _get_cache_file_path(self, value_key: str) -> str:
        """Return the path of the disk cache file for the given value."""
        cache_dir = get_cache_folder_path()
        return os.path.join(cache_dir, f'{self.function_key}-{value_key}.{_CACHED_FILE_EXTENSION}')

    def _is_cache_file(self, fname: str) -> bool:
        """Return true if the given file name is a cache file for this storage."""
        return fname.startswith(f'{self.function_key}-') and fname.endswith(f'.{_CACHED_FILE_EXTENSION}')