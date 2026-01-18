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
def _strided_from_memmap(filename, dtype, mode, offset, order, shape, strides, total_buffer_len, unlink_on_gc_collect):
    """Reconstruct an array view on a memory mapped file."""
    if mode == 'w+':
        mode = 'r+'
    if strides is None:
        return make_memmap(filename, dtype=dtype, shape=shape, mode=mode, offset=offset, order=order, unlink_on_gc_collect=unlink_on_gc_collect)
    else:
        base = make_memmap(filename, dtype=dtype, shape=total_buffer_len, offset=offset, mode=mode, order=order, unlink_on_gc_collect=unlink_on_gc_collect)
        return as_strided(base, shape=shape, strides=strides)