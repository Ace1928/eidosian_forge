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
def _contains_str_like(pat: Any) -> bool:
    """Determine if the object is a str-like or array of str-like."""
    if isinstance(pat, (str, bytes)):
        return True
    if not hasattr(pat, 'dtype'):
        return False
    return pat.dtype.kind in ['U', 'S']