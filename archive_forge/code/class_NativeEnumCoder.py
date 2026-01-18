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
class NativeEnumCoder(VariableCoder):
    """Encode Enum into variable dtype metadata."""

    def encode(self, variable: Variable, name: T_Name=None) -> Variable:
        if 'dtype' in variable.encoding and np.dtype(variable.encoding['dtype']).metadata and ('enum' in variable.encoding['dtype'].metadata):
            dims, data, attrs, encoding = unpack_for_encoding(variable)
            data = data.astype(dtype=variable.encoding.pop('dtype'))
            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable

    def decode(self, variable: Variable, name: T_Name=None) -> Variable:
        raise NotImplementedError()