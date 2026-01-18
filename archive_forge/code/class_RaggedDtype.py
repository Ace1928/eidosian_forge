from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@register_extension_dtype
class RaggedDtype(ExtensionDtype):
    """
    Pandas ExtensionDtype to represent a ragged array datatype

    Methods not otherwise documented here are inherited from ExtensionDtype;
    please see the corresponding method on that class for the docstring
    """
    type = np.ndarray
    base = np.dtype('O')
    _subtype_re = re.compile('^ragged\\[(?P<subtype>\\w+)\\]$')
    _metadata = ('_dtype',)

    @property
    def name(self):
        return 'Ragged[{subtype}]'.format(subtype=self.subtype)

    def __repr__(self):
        return self.name

    @classmethod
    def construct_array_type(cls):
        return RaggedArray

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError("'construct_from_string' expects a string, got %s" % type(string))
        string = string.lower()
        msg = "Cannot construct a 'RaggedDtype' from '{}'"
        if string.startswith('ragged'):
            try:
                subtype_string = cls._parse_subtype(string)
                return RaggedDtype(dtype=subtype_string)
            except Exception:
                raise TypeError(msg.format(string))
        else:
            raise TypeError(msg.format(string))

    def __init__(self, dtype=np.float64):
        if isinstance(dtype, RaggedDtype):
            self._dtype = dtype.subtype
        else:
            self._dtype = np.dtype(dtype)

    @property
    def subtype(self):
        return self._dtype

    @classmethod
    def _parse_subtype(cls, dtype_string):
        """
        Parse a datatype string to get the subtype

        Parameters
        ----------
        dtype_string: str
            A string like Ragged[subtype]

        Returns
        -------
        subtype: str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted
        """
        dtype_string = dtype_string.lower()
        match = cls._subtype_re.match(dtype_string)
        if match:
            subtype_string = match.groupdict()['subtype']
        elif dtype_string == 'ragged':
            subtype_string = 'float64'
        else:
            raise ValueError('Cannot parse {dtype_string}'.format(dtype_string=dtype_string))
        return subtype_string