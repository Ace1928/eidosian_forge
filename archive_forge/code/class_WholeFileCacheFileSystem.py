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
class WholeFileCacheFileSystem(CachingFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This is similar to ``CachingFileSystem``, but without
    the block-wise functionality and so can work even when sparse files
    are not allowed. See its docstring for definition of the init
    arguments.

    The class still needs access to the remote store for listing files,
    and may refresh cached files.
    """
    protocol = 'filecache'
    local_file = True

    def open_many(self, open_files, **kwargs):
        paths = [of.path for of in open_files]
        if 'r' in open_files.mode:
            self._mkcache()
        else:
            return [LocalTempFile(self.fs, path, mode=open_files.mode, fn=os.path.join(self.storage[-1], self._mapper(path)), **kwargs) for path in paths]
        if self.compression:
            raise NotImplementedError
        details = [self._check_file(sp) for sp in paths]
        downpath = [p for p, d in zip(paths, details) if not d]
        downfn0 = [os.path.join(self.storage[-1], self._mapper(p)) for p, d in zip(paths, details)]
        downfn = [fn for fn, d in zip(downfn0, details) if not d]
        if downpath:
            self.fs.get(downpath, downfn)
            newdetail = [{'original': path, 'fn': self._mapper(path), 'blocks': True, 'time': time.time(), 'uid': self.fs.ukey(path)} for path in downpath]
            for path, detail in zip(downpath, newdetail):
                self._metadata.update_file(path, detail)
            self.save_cache()

        def firstpart(fn):
            return fn[1] if isinstance(fn, tuple) else fn
        return [open(firstpart(fn0) if fn0 else fn1, mode=open_files.mode) for fn0, fn1 in zip(details, downfn0)]

    def commit_many(self, open_files):
        self.fs.put([f.fn for f in open_files], [f.path for f in open_files])
        [f.close() for f in open_files]
        for f in open_files:
            try:
                os.remove(f.name)
            except FileNotFoundError:
                pass
        self._cache_size = None

    def _make_local_details(self, path):
        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        detail = {'original': path, 'fn': hash, 'blocks': True, 'time': time.time(), 'uid': self.fs.ukey(path)}
        self._metadata.update_file(path, detail)
        logger.debug('Copying %s to local cache', path)
        return fn

    def cat(self, path, recursive=False, on_error='raise', callback=DEFAULT_CALLBACK, **kwargs):
        paths = self.expand_path(path, recursive=recursive, maxdepth=kwargs.get('maxdepth', None))
        getpaths = []
        storepaths = []
        fns = []
        out = {}
        for p in paths.copy():
            try:
                detail = self._check_file(p)
                if not detail:
                    fn = self._make_local_details(p)
                    getpaths.append(p)
                    storepaths.append(fn)
                else:
                    detail, fn = detail if isinstance(detail, tuple) else (None, detail)
                fns.append(fn)
            except Exception as e:
                if on_error == 'raise':
                    raise
                if on_error == 'return':
                    out[p] = e
                paths.remove(p)
        if getpaths:
            self.fs.get(getpaths, storepaths)
            self.save_cache()
        callback.set_size(len(paths))
        for p, fn in zip(paths, fns):
            with open(fn, 'rb') as f:
                out[p] = f.read()
            callback.relative_update(1)
        if isinstance(path, str) and len(paths) == 1 and (recursive is False):
            out = out[paths[0]]
        return out

    def _open(self, path, mode='rb', **kwargs):
        path = self._strip_protocol(path)
        if 'r' not in mode:
            fn = self._make_local_details(path)
            user_specified_kwargs = {k: v for k, v in kwargs.items() if k not in ['autocommit', 'block_size', 'cache_options']}
            return LocalTempFile(self, path, mode=mode, fn=fn, **user_specified_kwargs)
        detail = self._check_file(path)
        if detail:
            detail, fn = detail
            _, blocks = (detail['fn'], detail['blocks'])
            if blocks is True:
                logger.debug('Opening local copy of %s', path)
                f = open(fn, mode)
                f.original = detail.get('original')
                return f
            else:
                raise ValueError(f'Attempt to open partially cached file {path} as a wholly cached file')
        else:
            fn = self._make_local_details(path)
        kwargs['mode'] = mode
        self._mkcache()
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
        self.save_cache()
        return self._open(path, mode)