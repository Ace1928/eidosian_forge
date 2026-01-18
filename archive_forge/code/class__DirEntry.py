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
class _DirEntry(object):

    def __init__(self, resource_type, name):
        self.resource_type = resource_type
        self.name = name
        self._dir = OrderedDict()
        self._open_files = []
        self._bytes_file = None
        self.lock = RLock()
        current_time = time.time()
        self.created_time = current_time
        self.accessed_time = current_time
        self.modified_time = current_time
        if not self.is_dir:
            self._bytes_file = io.BytesIO()

    @property
    def bytes_file(self):
        return self._bytes_file

    @property
    def is_dir(self):
        return self.resource_type == ResourceType.directory

    @property
    def size(self):
        with self.lock:
            if self.is_dir:
                return 0
            else:
                _bytes_file = typing.cast(io.BytesIO, self._bytes_file)
                _bytes_file.seek(0, os.SEEK_END)
                return _bytes_file.tell()

    @overload
    def get_entry(self, name, default):
        pass

    @overload
    def get_entry(self, name):
        pass

    @overload
    def get_entry(self, name, default):
        pass

    def get_entry(self, name, default=None):
        assert self.is_dir, 'must be a directory'
        return self._dir.get(name, default)

    def set_entry(self, name, dir_entry):
        self._dir[name] = dir_entry

    def remove_entry(self, name):
        del self._dir[name]

    def clear(self):
        self._dir.clear()

    def __contains__(self, name):
        return name in self._dir

    def __len__(self):
        return len(self._dir)

    def list(self):
        return list(self._dir.keys())

    def add_open_file(self, memory_file):
        self._open_files.append(memory_file)

    def remove_open_file(self, memory_file):
        self._open_files.remove(memory_file)

    def to_info(self, namespaces=None):
        namespaces = namespaces or ()
        info = {'basic': {'name': self.name, 'is_dir': self.is_dir}}
        if 'details' in namespaces:
            info['details'] = {'_write': ['accessed', 'modified'], 'type': int(self.resource_type), 'size': self.size, 'accessed': self.accessed_time, 'modified': self.modified_time, 'created': self.created_time}
        return Info(info)