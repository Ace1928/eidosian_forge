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
class WriteCachedTransaction(Transaction):

    def complete(self, commit=True):
        rpaths = [f.path for f in self.files]
        lpaths = [f.fn for f in self.files]
        if commit:
            self.fs.put(lpaths, rpaths)
        self.files.clear()
        self.fs._intrans = False
        self.fs._transaction = None
        self.fs = None