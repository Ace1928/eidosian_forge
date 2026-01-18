from collections import Counter
import numpy as np
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
def _filter_attrs(attrs, ignored_attrs):
    """Return attrs that are not in ignored_attrs"""
    return {k: v for k, v in attrs.items() if k not in ignored_attrs}