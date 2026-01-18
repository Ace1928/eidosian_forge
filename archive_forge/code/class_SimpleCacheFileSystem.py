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
class SimpleCacheFileSystem(WholeFileCacheFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files, and
    does not keep any metadata about the download time or file details.
    It is therefore safer to use in multi-threaded/concurrent situations.

    This is the only of the caching filesystems that supports write: you will
    be given a real local open file, and upon close and commit, it will be
    uploaded to the target filesystem; the writability or the target URL is
    not checked until that time.

    """
    protocol = 'simplecache'
    local_file = True
    transaction_type = WriteCachedTransaction

    def __init__(self, **kwargs):
        kw = kwargs.copy()
        for key in ['cache_check', 'expiry_time', 'check_files']:
            kw[key] = False
        super().__init__(**kw)
        for storage in self.storage:
            if not os.path.exists(storage):
                os.makedirs(storage, exist_ok=True)

    def _check_file(self, path):
        self._check_cache()
        sha = self._mapper(path)
        for storage in self.storage:
            fn = os.path.join(storage, sha)
            if os.path.exists(fn):
                return fn

    def save_cache(self):
        pass

    def load_cache(self):
        pass

    def pipe_file(self, path, value=None, **kwargs):
        if self._intrans:
            with self.open(path, 'wb') as f:
                f.write(value)
        else:
            super().pipe_file(path, value)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        details = []
        try:
            details = self.fs.ls(path, detail=True, **kwargs).copy()
        except FileNotFoundError as e:
            ex = e
        else:
            ex = None
        if self._intrans:
            path1 = path.rstrip('/') + '/'
            for f in self.transaction.files:
                if f.path == path:
                    details.append({'name': path, 'size': f.size or f.tell(), 'type': 'file'})
                elif f.path.startswith(path1):
                    if f.path.count('/') == path1.count('/'):
                        details.append({'name': f.path, 'size': f.size or f.tell(), 'type': 'file'})
                    else:
                        dname = '/'.join(f.path.split('/')[:path1.count('/') + 1])
                        details.append({'name': dname, 'size': 0, 'type': 'directory'})
        if ex is not None and (not details):
            raise ex
        if detail:
            return details
        return sorted((_['name'] for _ in details))

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        if self._intrans:
            f = [_ for _ in self.transaction.files if _.path == path]
            if f:
                return {'name': path, 'size': f[0].size or f[0].tell(), 'type': 'file'}
            f = any((_.path.startswith(path + '/') for _ in self.transaction.files))
            if f:
                return {'name': path, 'size': 0, 'type': 'directory'}
        return self.fs.info(path, **kwargs)

    def pipe(self, path, value=None, **kwargs):
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError('path must be str or dict')

    def cat_ranges(self, paths, starts, ends, max_gap=None, on_error='return', **kwargs):
        lpaths = [self._check_file(p) for p in paths]
        rpaths = [p for l, p in zip(lpaths, paths) if l is False]
        lpaths = [l for l, p in zip(lpaths, paths) if l is False]
        self.fs.get(rpaths, lpaths)
        return super().cat_ranges(paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs)

    def _open(self, path, mode='rb', **kwargs):
        path = self._strip_protocol(path)
        sha = self._mapper(path)
        if 'r' not in mode:
            fn = os.path.join(self.storage[-1], sha)
            user_specified_kwargs = {k: v for k, v in kwargs.items() if k not in ['autocommit', 'block_size', 'cache_options']}
            return LocalTempFile(self, path, mode=mode, autocommit=not self._intrans, fn=fn, **user_specified_kwargs)
        fn = self._check_file(path)
        if fn:
            return open(fn, mode)
        fn = os.path.join(self.storage[-1], sha)
        logger.debug('Copying %s to local cache', path)
        kwargs['mode'] = mode
        self._mkcache()
        self._cache_size = None
        if self.compression:
            with self.fs._open(path, **kwargs) as f, open(fn, 'wb') as f2:
                if isinstance(f, AbstractBufferedFile):
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = infer_compression(path) if self.compression == 'infer' else self.compression
                f = compr[comp](f, mode='rb')
                data = True
                while data:
                    block = getattr(f, 'blocksize', 5 * 2 ** 20)
                    data = f.read(block)
                    f2.write(data)
        else:
            self.fs.get_file(path, fn)
        return self._open(path, mode)