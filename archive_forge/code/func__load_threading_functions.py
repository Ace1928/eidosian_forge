import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
def _load_threading_functions(lib):
    ll.add_symbol('get_num_threads', lib.get_num_threads)
    ll.add_symbol('set_num_threads', lib.set_num_threads)
    ll.add_symbol('get_thread_id', lib.get_thread_id)
    global _set_num_threads
    _set_num_threads = CFUNCTYPE(None, c_int)(lib.set_num_threads)
    _set_num_threads(NUM_THREADS)
    global _get_num_threads
    _get_num_threads = CFUNCTYPE(c_int)(lib.get_num_threads)
    global _get_thread_id
    _get_thread_id = CFUNCTYPE(c_int)(lib.get_thread_id)
    ll.add_symbol('set_parallel_chunksize', lib.set_parallel_chunksize)
    ll.add_symbol('get_parallel_chunksize', lib.get_parallel_chunksize)
    ll.add_symbol('get_sched_size', lib.get_sched_size)
    global _set_parallel_chunksize
    _set_parallel_chunksize = CFUNCTYPE(c_uint, c_uint)(lib.set_parallel_chunksize)
    global _get_parallel_chunksize
    _get_parallel_chunksize = CFUNCTYPE(c_uint)(lib.get_parallel_chunksize)
    global _get_sched_size
    _get_sched_size = CFUNCTYPE(c_uint, c_uint, c_uint, POINTER(c_int), POINTER(c_int))(lib.get_sched_size)