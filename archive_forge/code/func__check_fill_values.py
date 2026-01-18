from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _check_fill_values(attrs, name, dtype):
    """ "Check _FillValue and missing_value if available.

    Return dictionary with raw fill values and set with encoded fill values.

    Issue SerializationWarning if appropriate.
    """
    raw_fill_dict = {}
    [pop_to(attrs, raw_fill_dict, attr, name=name) for attr in ('missing_value', '_FillValue')]
    encoded_fill_values = set()
    for k in list(raw_fill_dict):
        v = raw_fill_dict[k]
        kfill = {fv for fv in np.ravel(v) if not pd.isnull(fv)}
        if not kfill and np.issubdtype(dtype, np.integer):
            warnings.warn(f'variable {name!r} has non-conforming {k!r} {v!r} defined, dropping {k!r} entirely.', SerializationWarning, stacklevel=3)
            del raw_fill_dict[k]
        else:
            encoded_fill_values |= kfill
        if len(encoded_fill_values) > 1:
            warnings.warn(f'variable {name!r} has multiple fill values {encoded_fill_values} defined, decoding all values to NaN.', SerializationWarning, stacklevel=3)
    return (raw_fill_dict, encoded_fill_values)