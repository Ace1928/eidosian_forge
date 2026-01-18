import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _iter_member_by_value_(cls, value):
    """
        Extract all members from the value in definition (i.e. increasing value) order.
        """
    for val in _iter_bits_lsb(value & cls._flag_mask_):
        yield cls._value2member_map_.get(val)