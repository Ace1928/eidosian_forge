from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def _outer_to_numpy_indexer(indexer: BasicIndexer | OuterIndexer, shape: _Shape):
    """Convert an OuterIndexer into an indexer for NumPy.

    Parameters
    ----------
    indexer : Basic/OuterIndexer
        An indexer to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    tuple
        Tuple suitable for use to index a NumPy array.
    """
    if len([k for k in indexer.tuple if not isinstance(k, slice)]) <= 1:
        return indexer.tuple
    else:
        return _outer_to_vectorized_indexer(indexer, shape).tuple