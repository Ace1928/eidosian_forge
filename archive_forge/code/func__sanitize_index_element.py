from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
def _sanitize_index_element(ind):
    """Sanitize a one-element index."""
    if isinstance(ind, Number):
        ind2 = int(ind)
        if ind2 != ind:
            raise IndexError('Bad index.  Must be integer-like: %s' % ind)
        else:
            return ind2
    elif ind is None:
        return None
    elif is_dask_collection(ind):
        if ind.dtype.kind != 'i' or ind.size != 1:
            raise IndexError(f'Bad index. Must be integer-like: {ind}')
        return ind
    else:
        raise TypeError('Invalid index type', type(ind), ind)