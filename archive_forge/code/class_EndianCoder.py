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
class EndianCoder(VariableCoder):
    """Decode Endianness to native."""

    def encode(self):
        raise NotImplementedError()

    def decode(self, variable: Variable, name: T_Name=None) -> Variable:
        dims, data, attrs, encoding = unpack_for_decoding(variable)
        if not data.dtype.isnative:
            data = NativeEndiannessArray(data)
            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable