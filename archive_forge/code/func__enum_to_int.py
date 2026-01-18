import _signal
from _signal import *
from enum import IntEnum as _IntEnum
def _enum_to_int(value):
    """Convert an IntEnum member to a numeric value.
    If it's not an IntEnum member return the value itself.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return value