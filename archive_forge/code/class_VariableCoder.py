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
class VariableCoder:
    """Base class for encoding and decoding transformations on variables.

    We use coders for transforming variables between xarray's data model and
    a format suitable for serialization. For example, coders apply CF
    conventions for how data should be represented in netCDF files.

    Subclasses should implement encode() and decode(), which should satisfy
    the identity ``coder.decode(coder.encode(variable)) == variable``. If any
    options are necessary, they should be implemented as arguments to the
    __init__ method.

    The optional name argument to encode() and decode() exists solely for the
    sake of better error messages, and should correspond to the name of
    variables in the underlying store.
    """

    def encode(self, variable: Variable, name: T_Name=None) -> Variable:
        """Convert an encoded variable to a decoded variable"""
        raise NotImplementedError()

    def decode(self, variable: Variable, name: T_Name=None) -> Variable:
        """Convert an decoded variable to a encoded variable"""
        raise NotImplementedError()