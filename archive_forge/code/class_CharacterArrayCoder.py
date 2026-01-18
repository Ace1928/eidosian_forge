from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class CharacterArrayCoder(VariableCoder):
    """Transforms between arrays containing bytes and character arrays."""

    def encode(self, variable, name=None):
        variable = ensure_fixed_length_bytes(variable)
        dims, data, attrs, encoding = unpack_for_encoding(variable)
        if data.dtype.kind == 'S' and encoding.get('dtype') is not str:
            data = bytes_to_char(data)
            if 'char_dim_name' in encoding.keys():
                char_dim_name = encoding.pop('char_dim_name')
            else:
                char_dim_name = f'string{data.shape[-1]}'
            dims = dims + (char_dim_name,)
        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)
        if data.dtype == 'S1' and dims:
            encoding['char_dim_name'] = dims[-1]
            dims = dims[:-1]
            data = char_to_bytes(data)
        return Variable(dims, data, attrs, encoding)