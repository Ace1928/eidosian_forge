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
def _reduce_memmap_backed(a, m):
    """Pickling reduction for memmap backed arrays.

    a is expected to be an instance of np.ndarray (or np.memmap)
    m is expected to be an instance of np.memmap on the top of the ``base``
    attribute ancestry of a. ``m.base`` should be the real python mmap object.
    """
    util.debug('[MEMMAP REDUCE] reducing a memmap-backed array (shape, {}, pid: {})'.format(a.shape, os.getpid()))
    a_start, a_end = np.byte_bounds(a)
    m_start = np.byte_bounds(m)[0]
    offset = a_start - m_start
    offset += m.offset
    if m.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        order = 'C'
    if a.flags['F_CONTIGUOUS'] or a.flags['C_CONTIGUOUS']:
        strides = None
        total_buffer_len = None
    else:
        strides = a.strides
        total_buffer_len = (a_end - a_start) // a.itemsize
    return (_strided_from_memmap, (m.filename, a.dtype, m.mode, offset, order, a.shape, strides, total_buffer_len, False))