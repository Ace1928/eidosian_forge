from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
def _get_sizeof_on_path(path, size):
    sys.path.append(os.fsdecode(path))
    import dask.sizeof
    dask.sizeof._register_entry_point_plugins()
    import class_impl
    cls = class_impl.Impl(size)
    return sizeof(cls)