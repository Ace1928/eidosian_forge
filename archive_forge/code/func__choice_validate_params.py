from __future__ import annotations
import contextlib
import importlib
import numbers
from itertools import chain, product
from numbers import Integral
from operator import getitem
from threading import Lock
import numpy as np
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.creation import arange
from dask.array.utils import asarray_safe
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, random_state_data, typename
def _choice_validate_params(state, a, size, replace, p, axis, chunks):
    dependencies = []
    if isinstance(a, Integral):
        if isinstance(state, Generator):
            if state._backend_name == 'cupy':
                raise NotImplementedError('`choice` not supported for cupy-backed `Generator`.')
            meta = state._backend.random.default_rng().choice(1, size=(), p=None)
        elif isinstance(state, RandomState):
            dummy_p = state._backend.array([1]) if p is not None else p
            meta = state._backend.random.RandomState().choice(1, size=(), p=dummy_p)
        else:
            raise ValueError('Unknown generator class')
        len_a = a
        if a < 0:
            raise ValueError('a must be greater than 0')
    else:
        a = asarray(a)
        a = a.rechunk(a.shape)
        meta = a._meta
        if a.ndim != 1:
            raise ValueError('a must be one dimensional')
        len_a = len(a)
        dependencies.append(a)
        a = a.__dask_keys__()[0]
    if p is not None:
        if not isinstance(p, Array):
            p = asarray_safe(p, like=p)
            if not np.isclose(p.sum(), 1, rtol=1e-07, atol=0):
                raise ValueError('probabilities do not sum to 1')
            p = asarray(p)
        else:
            p = p.rechunk(p.shape)
        if p.ndim != 1:
            raise ValueError('p must be one dimensional')
        if len(p) != len_a:
            raise ValueError('a and p must have the same size')
        dependencies.append(p)
        p = p.__dask_keys__()[0]
    if size is None:
        size = ()
    elif not isinstance(size, (tuple, list)):
        size = (size,)
    if axis != 0:
        raise ValueError('axis must be 0 since a is one dimensional')
    chunks = normalize_chunks(chunks, size, dtype=np.float64)
    if not replace and len(chunks[0]) > 1:
        err_msg = 'replace=False is not currently supported for dask.array.choice with multi-chunk output arrays'
        raise NotImplementedError(err_msg)
    return (a, size, replace, p, axis, chunks, meta, dependencies)