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