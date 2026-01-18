from __future__ import annotations
import inspect
import logging
import os
import tempfile
import time
import weakref
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Callable, ClassVar
from fsspec import AbstractFileSystem, filesystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.spec import AbstractBufferedFile
from fsspec.transaction import Transaction
from fsspec.utils import infer_compression
def _make_local_details(self, path):
    hash = self._mapper(path)
    fn = os.path.join(self.storage[-1], hash)
    detail = {'original': path, 'fn': hash, 'blocks': True, 'time': time.time(), 'uid': self.fs.ukey(path)}
    self._metadata.update_file(path, detail)
    logger.debug('Copying %s to local cache', path)
    return fn