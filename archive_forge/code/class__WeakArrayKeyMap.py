from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
class _WeakArrayKeyMap:
    """A variant of weakref.WeakKeyDictionary for unhashable numpy arrays.

    This datastructure will be used with numpy arrays as obj keys, therefore we
    do not use the __get__ / __set__ methods to avoid any conflict with the
    numpy fancy indexing syntax.
    """

    def __init__(self):
        self._data = {}

    def get(self, obj):
        ref, val = self._data[id(obj)]
        if ref() is not obj:
            raise KeyError(obj)
        return val

    def set(self, obj, value):
        key = id(obj)
        try:
            ref, _ = self._data[key]
            if ref() is not obj:
                raise KeyError(obj)
        except KeyError:

            def on_destroy(_):
                del self._data[key]
            ref = weakref.ref(obj, on_destroy)
        self._data[key] = (ref, value)

    def __getstate__(self):
        raise PicklingError('_WeakArrayKeyMap is not pickleable')