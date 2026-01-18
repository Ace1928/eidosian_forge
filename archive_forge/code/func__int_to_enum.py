import _signal
from _signal import *
from enum import IntEnum as _IntEnum
def _int_to_enum(value, enum_klass):
    """Convert a possible numeric value to an IntEnum member.
    If it's not a known member, return the value itself.
    """
    if not isinstance(value, int):
        return value
    try:
        return enum_klass(value)
    except ValueError:
        return value