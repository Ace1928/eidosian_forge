from collections import Counter
import numpy as np
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
def _iris_obj_to_attrs(obj):
    """Return a dictionary of attrs when given a Iris object"""
    attrs = {'standard_name': obj.standard_name, 'long_name': obj.long_name}
    if obj.units.calendar:
        attrs['calendar'] = obj.units.calendar
    if obj.units.origin != '1' and (not obj.units.is_unknown()):
        attrs['units'] = obj.units.origin
    attrs.update(obj.attributes)
    return {k: v for k, v in attrs.items() if v is not None}