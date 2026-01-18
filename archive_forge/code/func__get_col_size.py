from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
def _get_col_size(data):
    p = 0
    if not isinstance(data, pa.ChunkedArray):
        data = data.data
    for chunk in data.iterchunks():
        for buffer in chunk.buffers():
            if buffer:
                p += buffer.size
    return p