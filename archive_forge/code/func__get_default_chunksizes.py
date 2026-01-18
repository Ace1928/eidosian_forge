import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def _get_default_chunksizes(dimsizes, dtype):
    CHUNK_BASE = 16 * 1024
    CHUNK_MIN = 8 * 1024
    CHUNK_MAX = 1024 * 1024
    type_size = np.dtype(dtype).itemsize
    is_unlimited = np.array([x == 0 for x in dimsizes])
    chunks = np.array([x if x != 0 else 1024 for x in dimsizes], dtype='=f8')
    ndims = len(dimsizes)
    if ndims == 0:
        raise ValueError('Chunks not allowed for scalar datasets.')
    if not np.all(np.isfinite(chunks)):
        raise ValueError('Illegal value in chunk tuple')
    dset_size = np.prod(chunks[~is_unlimited]) * type_size
    target_size = CHUNK_BASE * 2 ** np.log10(dset_size / (1024 * 1024))
    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN
    i = 0
    while True:
        idx = i % ndims
        chunk_bytes = np.prod(chunks) * type_size
        done = (chunk_bytes < target_size or abs(chunk_bytes - target_size) / target_size < 0.5) and chunk_bytes < CHUNK_MAX
        if done:
            break
        if np.prod(chunks) == 1:
            break
        nelem_unlim = np.prod(chunks[is_unlimited])
        if nelem_unlim == 1 or is_unlimited[idx]:
            chunks[idx] = np.ceil(chunks[idx] / 2.0)
        i += 1
    return tuple((int(x) for x in chunks))