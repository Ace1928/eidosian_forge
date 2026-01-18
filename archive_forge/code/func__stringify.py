from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
def _stringify(self, invar: Any) -> str | bytes | Any:
    """
        Convert a string-like to the correct string/bytes type.

        This is mostly here to tell mypy a pattern is a str/bytes not a re.Pattern.
        """
    if hasattr(invar, 'astype'):
        return invar.astype(self._obj.dtype.kind)
    else:
        return self._obj.dtype.type(invar)