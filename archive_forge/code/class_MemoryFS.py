from __future__ import absolute_import, unicode_literals
import typing
import contextlib
import io
import os
import six
import time
from collections import OrderedDict
from threading import RLock
from . import errors
from ._typing import overload
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType, Seek
from .info import Info
from .mode import Mode
from .path import iteratepath, normpath, split
@six.python_2_unicode_compatible
class MemoryFS(FS):
    """A filesystem that stored in memory.

    Memory filesystems are useful for caches, temporary data stores,
    unit testing, etc. Since all the data is in memory, they are very
    fast, but non-permanent. The `MemoryFS` constructor takes no
    arguments.

    Examples:
        Create with the constructor::

            >>> from fs.memoryfs import MemoryFS
            >>> mem_fs = MemoryFS()

        Or via an FS URL::

            >>> import fs
            >>> mem_fs = fs.open_fs('mem://')

    """
    _meta = {'case_insensitive': False, 'invalid_path_chars': '\x00', 'network': False, 'read_only': False, 'thread_safe': True, 'unicode_paths': True, 'virtual': False}

    def __init__(self):
        """Create an in-memory filesystem."""
        self._meta = self._meta.copy()
        self.root = self._make_dir_entry(ResourceType.directory, '')
        super(MemoryFS, self).__init__()

    def __repr__(self):
        return 'MemoryFS()'

    def __str__(self):
        return '<memfs>'

    def _make_dir_entry(self, resource_type, name):
        return _DirEntry(resource_type, name)

    def _get_dir_entry(self, dir_path):
        """Get a directory entry, or `None` if one doesn't exist."""
        with self._lock:
            dir_path = normpath(dir_path)
            current_entry = self.root
            for path_component in iteratepath(dir_path):
                if current_entry is None:
                    return None
                if not current_entry.is_dir:
                    return None
                current_entry = current_entry.get_entry(path_component)
            return current_entry

    def close(self):
        if not self._closed:
            del self.root
        return super(MemoryFS, self).close()

    def getinfo(self, path, namespaces=None):
        _path = self.validatepath(path)
        dir_entry = self._get_dir_entry(_path)
        if dir_entry is None:
            raise errors.ResourceNotFound(path)
        return dir_entry.to_info(namespaces=namespaces)

    def listdir(self, path):
        self.check()
        _path = self.validatepath(path)
        with self._lock:
            dir_entry = self._get_dir_entry(_path)
            if dir_entry is None:
                raise errors.ResourceNotFound(path)
            if not dir_entry.is_dir:
                raise errors.DirectoryExpected(path)
            return dir_entry.list()
    if typing.TYPE_CHECKING:

        def opendir(self, path, factory=None):
            pass

    def makedir(self, path, permissions=None, recreate=False):
        _path = self.validatepath(path)
        with self._lock:
            if _path == '/':
                if recreate:
                    return self.opendir(path)
                else:
                    raise errors.DirectoryExists(path)
            dir_path, dir_name = split(_path)
            parent_dir = self._get_dir_entry(dir_path)
            if parent_dir is None:
                raise errors.ResourceNotFound(path)
            dir_entry = parent_dir.get_entry(dir_name)
            if dir_entry is not None and (not recreate):
                raise errors.DirectoryExists(path)
            if dir_entry is None:
                new_dir = self._make_dir_entry(ResourceType.directory, dir_name)
                parent_dir.set_entry(dir_name, new_dir)
            return self.opendir(path)

    def move(self, src_path, dst_path, overwrite=False, preserve_time=False):
        src_dir, src_name = split(self.validatepath(src_path))
        dst_dir, dst_name = split(self.validatepath(dst_path))
        with self._lock:
            src_dir_entry = self._get_dir_entry(src_dir)
            if src_dir_entry is None or src_name not in src_dir_entry:
                raise errors.ResourceNotFound(src_path)
            src_entry = src_dir_entry.get_entry(src_name)
            if src_entry.is_dir:
                raise errors.FileExpected(src_path)
            dst_dir_entry = self._get_dir_entry(dst_dir)
            if dst_dir_entry is None:
                raise errors.ResourceNotFound(dst_path)
            elif not overwrite and dst_name in dst_dir_entry:
                raise errors.DestinationExists(dst_path)
            dst_dir_entry.set_entry(dst_name, src_entry)
            src_dir_entry.remove_entry(src_name)
            src_entry.name = dst_name
            if preserve_time:
                copy_modified_time(self, src_path, self, dst_path)

    def movedir(self, src_path, dst_path, create=False, preserve_time=False):
        src_dir, src_name = split(self.validatepath(src_path))
        dst_dir, dst_name = split(self.validatepath(dst_path))
        with self._lock:
            src_dir_entry = self._get_dir_entry(src_dir)
            if src_dir_entry is None or src_name not in src_dir_entry:
                raise errors.ResourceNotFound(src_path)
            src_entry = src_dir_entry.get_entry(src_name)
            if not src_entry.is_dir:
                raise errors.DirectoryExpected(src_path)
            dst_dir_entry = self._get_dir_entry(dst_dir)
            if dst_dir_entry is None or (not create and dst_name not in dst_dir_entry):
                raise errors.ResourceNotFound(dst_path)
            dst_dir_entry.set_entry(dst_name, src_entry)
            src_dir_entry.remove_entry(src_name)
            src_entry.name = dst_name
            if preserve_time:
                copy_modified_time(self, src_path, self, dst_path)

    def openbin(self, path, mode='r', buffering=-1, **options):
        _mode = Mode(mode)
        _mode.validate_bin()
        _path = self.validatepath(path)
        dir_path, file_name = split(_path)
        if not file_name:
            raise errors.FileExpected(path)
        with self._lock:
            parent_dir_entry = self._get_dir_entry(dir_path)
            if parent_dir_entry is None or not parent_dir_entry.is_dir:
                raise errors.ResourceNotFound(path)
            if _mode.create:
                if file_name not in parent_dir_entry:
                    file_dir_entry = self._make_dir_entry(ResourceType.file, file_name)
                    parent_dir_entry.set_entry(file_name, file_dir_entry)
                else:
                    file_dir_entry = self._get_dir_entry(_path)
                    if _mode.exclusive:
                        raise errors.FileExists(path)
                if file_dir_entry.is_dir:
                    raise errors.FileExpected(path)
                mem_file = _MemoryFile(path=_path, memory_fs=self, mode=mode, dir_entry=file_dir_entry)
                file_dir_entry.add_open_file(mem_file)
                return mem_file
            if file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)
            file_dir_entry = parent_dir_entry.get_entry(file_name)
            if file_dir_entry.is_dir:
                raise errors.FileExpected(path)
            mem_file = _MemoryFile(path=_path, memory_fs=self, mode=mode, dir_entry=file_dir_entry)
            file_dir_entry.add_open_file(mem_file)
            return mem_file

    def remove(self, path):
        _path = self.validatepath(path)
        with self._lock:
            dir_path, file_name = split(_path)
            parent_dir_entry = self._get_dir_entry(dir_path)
            if parent_dir_entry is None or file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)
            file_dir_entry = typing.cast(_DirEntry, self._get_dir_entry(_path))
            if file_dir_entry.is_dir:
                raise errors.FileExpected(path)
            parent_dir_entry.remove_entry(file_name)

    def removedir(self, path):
        _path = self.validatepath(path)
        if _path == '/':
            raise errors.RemoveRootError()
        if not self.isempty(path):
            raise errors.DirectoryNotEmpty(path)
        self.removetree(_path)

    def removetree(self, path):
        _path = self.validatepath(path)
        with self._lock:
            if _path == '/':
                self.root.clear()
                return
            dir_path, file_name = split(_path)
            parent_dir_entry = self._get_dir_entry(dir_path)
            if parent_dir_entry is None or file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)
            dir_dir_entry = typing.cast(_DirEntry, self._get_dir_entry(_path))
            if not dir_dir_entry.is_dir:
                raise errors.DirectoryExpected(path)
            parent_dir_entry.remove_entry(file_name)

    def scandir(self, path, namespaces=None, page=None):
        self.check()
        _path = self.validatepath(path)
        with self._lock:
            dir_entry = self._get_dir_entry(_path)
            if dir_entry is None:
                raise errors.ResourceNotFound(path)
            if not dir_entry.is_dir:
                raise errors.DirectoryExpected(path)
            filenames = dir_entry.list()
            if page is not None:
                start, end = page
                filenames = filenames[start:end]
            for name in filenames:
                entry = typing.cast(_DirEntry, dir_entry.get_entry(name))
                yield entry.to_info(namespaces=namespaces)

    def setinfo(self, path, info):
        _path = self.validatepath(path)
        with self._lock:
            dir_path, file_name = split(_path)
            parent_dir_entry = self._get_dir_entry(dir_path)
            if parent_dir_entry is None or file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)
            resource_entry = typing.cast(_DirEntry, parent_dir_entry.get_entry(file_name))
            if 'details' in info:
                details = info['details']
                if 'accessed' in details:
                    resource_entry.accessed_time = details['accessed']
                if 'modified' in details:
                    resource_entry.modified_time = details['modified']