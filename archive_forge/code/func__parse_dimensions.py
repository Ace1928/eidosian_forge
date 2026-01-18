from __future__ import annotations
import copy
import math
import sys
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import (
import numpy as np
from xarray.core import dtypes, formatting, formatting_html
from xarray.core.indexing import (
from xarray.namedarray._aggregations import NamedArrayAggregations
from xarray.namedarray._typing import (
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import to_numpy
from xarray.namedarray.utils import (
def _parse_dimensions(self, dims: _DimsLike) -> _Dims:
    dims = (dims,) if isinstance(dims, str) else tuple(dims)
    if len(dims) != self.ndim:
        raise ValueError(f'dimensions {dims} must have the same length as the number of data dimensions, ndim={self.ndim}')
    if len(set(dims)) < len(dims):
        repeated_dims = {d for d in dims if dims.count(d) > 1}
        warnings.warn(f"Duplicate dimension names present: dimensions {repeated_dims} appear more than once in dims={dims}. We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.", UserWarning)
    return dims