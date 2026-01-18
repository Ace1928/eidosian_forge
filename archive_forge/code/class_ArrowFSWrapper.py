import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
class ArrowFSWrapper(AbstractFileSystem):
    """FSSpec-compatible wrapper of pyarrow.fs.FileSystem.

    Parameters
    ----------
    fs : pyarrow.fs.FileSystem

    """
    root_marker = '/'

    def __init__(self, fs, **kwargs):
        global PYARROW_VERSION
        PYARROW_VERSION = get_package_version_without_import('pyarrow')
        self.fs = fs
        super().__init__(**kwargs)

    @property
    def protocol(self):
        return self.fs.type_name

    @cached_property
    def fsid(self):
        return 'hdfs_' + tokenize(self.fs.host, self.fs.port)

    @classmethod
    def _strip_protocol(cls, path):
        ops = infer_storage_options(path)
        path = ops['path']
        if path.startswith('//'):
            path = path[1:]
        return path

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        from pyarrow.fs import FileSelector
        entries = [self._make_entry(entry) for entry in self.fs.get_file_info(FileSelector(path))]
        if detail:
            return entries
        else:
            return [entry['name'] for entry in entries]

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        [info] = self.fs.get_file_info([path])
        return self._make_entry(info)

    def exists(self, path):
        path = self._strip_protocol(path)
        try:
            self.info(path)
        except FileNotFoundError:
            return False
        else:
            return True

    def _make_entry(self, info):
        from pyarrow.fs import FileType
        if info.type is FileType.Directory:
            kind = 'directory'
        elif info.type is FileType.File:
            kind = 'file'
        elif info.type is FileType.NotFound:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), info.path)
        else:
            kind = 'other'
        return {'name': info.path, 'size': info.size, 'type': kind, 'mtime': info.mtime}

    @wrap_exceptions
    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip('/')
        path2 = self._strip_protocol(path2).rstrip('/')
        with self._open(path1, 'rb') as lstream:
            tmp_fname = f'{path2}.tmp.{secrets.token_hex(6)}'
            try:
                with self.open(tmp_fname, 'wb') as rstream:
                    shutil.copyfileobj(lstream, rstream)
                self.fs.move(tmp_fname, path2)
            except BaseException:
                with suppress(FileNotFoundError):
                    self.fs.delete_file(tmp_fname)
                raise

    @wrap_exceptions
    def mv(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip('/')
        path2 = self._strip_protocol(path2).rstrip('/')
        self.fs.move(path1, path2)
    mv_file = mv

    @wrap_exceptions
    def rm_file(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_file(path)

    @wrap_exceptions
    def rm(self, path, recursive=False, maxdepth=None):
        path = self._strip_protocol(path).rstrip('/')
        if self.isdir(path):
            if recursive:
                self.fs.delete_dir(path)
            else:
                raise ValueError("Can't delete directories without recursive=False")
        else:
            self.fs.delete_file(path)

    @wrap_exceptions
    def _open(self, path, mode='rb', block_size=None, seekable=True, **kwargs):
        if mode == 'rb':
            if seekable:
                method = self.fs.open_input_file
            else:
                method = self.fs.open_input_stream
        elif mode == 'wb':
            method = self.fs.open_output_stream
        elif mode == 'ab':
            method = self.fs.open_append_stream
        else:
            raise ValueError(f'unsupported mode for Arrow filesystem: {mode!r}')
        _kwargs = {}
        if mode != 'rb' or not seekable:
            if int(PYARROW_VERSION.split('.')[0]) >= 4:
                _kwargs['compression'] = None
        stream = method(path, **_kwargs)
        return ArrowFile(self, stream, path, mode, block_size, **kwargs)

    @wrap_exceptions
    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if create_parents:
            self.makedirs(path, exist_ok=True)
        else:
            self.fs.create_dir(path, recursive=False)

    @wrap_exceptions
    def makedirs(self, path, exist_ok=False):
        path = self._strip_protocol(path)
        self.fs.create_dir(path, recursive=True)

    @wrap_exceptions
    def rmdir(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_dir(path)

    @wrap_exceptions
    def modified(self, path):
        path = self._strip_protocol(path)
        return self.fs.get_file_info(path).mtime

    def cat_file(self, path, start=None, end=None, **kwargs):
        kwargs['seekable'] = start not in [None, 0]
        return super().cat_file(path, start=None, end=None, **kwargs)

    def get_file(self, rpath, lpath, **kwargs):
        kwargs['seekable'] = False
        super().get_file(rpath, lpath, **kwargs)