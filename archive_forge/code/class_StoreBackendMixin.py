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
class StoreBackendMixin(object):
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of
           strings."""
        full_path = os.path.join(self.location, *path)
        if verbose > 1:
            if verbose < 10:
                print('{0}...'.format(msg))
            else:
                print('{0} from {1}'.format(msg, full_path))
        mmap_mode = None if not hasattr(self, 'mmap_mode') else self.mmap_mode
        filename = os.path.join(full_path, 'output.pkl')
        if not self._item_exists(filename):
            raise KeyError('Non-existing item (may have been cleared).\nFile %s does not exist' % filename)
        if mmap_mode is None:
            with self._open_item(filename, 'rb') as f:
                item = numpy_pickle.load(f)
        else:
            item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return item

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
           strings."""
        try:
            item_path = os.path.join(self.location, *path)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, 'output.pkl')
            if verbose > 10:
                print('Persisting in %s' % item_path)

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, 'wb') as f:
                    try:
                        numpy_pickle.dump(to_write, f, compress=self.compress)
                    except PicklingError as e:
                        warnings.warn(f'Unable to cache to disk: failed to pickle output. In version 1.5 this will raise an exception. Exception: {e}.', FutureWarning)
            self._concurrency_safe_write(item, filename, write_func)
        except Exception as e:
            warnings.warn(f'Unable to cache to disk. Possibly a race condition in the creation of the directory. Exception: {e}.', CacheWarning)

    def clear_item(self, path):
        """Clear the item at the path, given as a list of strings."""
        item_path = os.path.join(self.location, *path)
        if self._item_exists(item_path):
            self.clear_location(item_path)

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of
           strings"""
        item_path = os.path.join(self.location, *path)
        filename = os.path.join(item_path, 'output.pkl')
        return self._item_exists(filename)

    def get_item_info(self, path):
        """Return information about item."""
        return {'location': os.path.join(self.location, *path)}

    def get_metadata(self, path):
        """Return actual metadata of an item."""
        try:
            item_path = os.path.join(self.location, *path)
            filename = os.path.join(item_path, 'metadata.json')
            with self._open_item(filename, 'rb') as f:
                return json.loads(f.read().decode('utf-8'))
        except:
            return {}

    def store_metadata(self, path, metadata):
        """Store metadata of a computation."""
        try:
            item_path = os.path.join(self.location, *path)
            self.create_location(item_path)
            filename = os.path.join(item_path, 'metadata.json')

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, 'wb') as f:
                    f.write(json.dumps(to_write).encode('utf-8'))
            self._concurrency_safe_write(metadata, filename, write_func)
        except:
            pass

    def contains_path(self, path):
        """Check cached function is available in store."""
        func_path = os.path.join(self.location, *path)
        return self.object_exists(func_path)

    def clear_path(self, path):
        """Clear all items with a common path in the store."""
        func_path = os.path.join(self.location, *path)
        if self._item_exists(func_path):
            self.clear_location(func_path)

    def store_cached_func_code(self, path, func_code=None):
        """Store the code of the cached function."""
        func_path = os.path.join(self.location, *path)
        if not self._item_exists(func_path):
            self.create_location(func_path)
        if func_code is not None:
            filename = os.path.join(func_path, 'func_code.py')
            with self._open_item(filename, 'wb') as f:
                f.write(func_code.encode('utf-8'))

    def get_cached_func_code(self, path):
        """Store the code of the cached function."""
        path += ['func_code.py']
        filename = os.path.join(self.location, *path)
        try:
            with self._open_item(filename, 'rb') as f:
                return f.read().decode('utf-8')
        except:
            raise

    def get_cached_func_info(self, path):
        """Return information related to the cached function if it exists."""
        return {'location': os.path.join(self.location, *path)}

    def clear(self):
        """Clear the whole store content."""
        self.clear_location(self.location)

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Remove the store's oldest files to enforce item, byte, and age limits.
        """
        items_to_delete = self._get_items_to_delete(bytes_limit, items_limit, age_limit)
        for item in items_to_delete:
            if self.verbose > 10:
                print('Deleting item {0}'.format(item))
            try:
                self.clear_location(item.path)
            except OSError:
                pass

    def _get_items_to_delete(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Get items to delete to keep the store under size, file, & age limits.
        """
        if isinstance(bytes_limit, str):
            bytes_limit = memstr_to_bytes(bytes_limit)
        items = self.get_items()
        size = sum((item.size for item in items))
        if bytes_limit is not None:
            to_delete_size = size - bytes_limit
        else:
            to_delete_size = 0
        if items_limit is not None:
            to_delete_items = len(items) - items_limit
        else:
            to_delete_items = 0
        if age_limit is not None:
            older_item = min((item.last_access for item in items))
            deadline = datetime.datetime.now() - age_limit
        else:
            deadline = None
        if to_delete_size <= 0 and to_delete_items <= 0 and (deadline is None or older_item > deadline):
            return []
        items.sort(key=operator.attrgetter('last_access'))
        items_to_delete = []
        size_so_far = 0
        items_so_far = 0
        for item in items:
            if size_so_far >= to_delete_size and items_so_far >= to_delete_items and (deadline is None or deadline < item.last_access):
                break
            items_to_delete.append(item)
            size_so_far += item.size
            items_so_far += 1
        return items_to_delete

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        temporary_filename = concurrency_safe_write(to_write, filename, write_func)
        self._move_item(temporary_filename, filename)

    def __repr__(self):
        """Printable representation of the store location."""
        return '{class_name}(location="{location}")'.format(class_name=self.__class__.__name__, location=self.location)