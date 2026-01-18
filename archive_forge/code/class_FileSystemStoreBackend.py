from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from . import numpy_pickle
class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""
    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []
        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match('[a-f0-9]{32}', os.path.basename(dirpath))
            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, 'output.pkl')
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        continue
                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum((os.path.getsize(fn) for fn in full_filenames))
                except OSError:
                    continue
                items.append(CacheItemInfo(dirpath, dirsize, last_access))
        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        if backend_options is None:
            backend_options = {}
        self.location = location
        if not os.path.exists(self.location):
            mkdirp(self.location)
        self.compress = backend_options.get('compress', False)
        mmap_mode = backend_options.get('mmap_mode')
        if self.compress and mmap_mode is not None:
            warnings.warn('Compressed items cannot be memmapped in a filesystem store. Option will be ignored.', stacklevel=2)
        self.mmap_mode = mmap_mode
        self.verbose = verbose