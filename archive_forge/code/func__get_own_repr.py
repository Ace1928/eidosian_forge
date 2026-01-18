import ctypes, ctypes.util, operator, sys
from . import model
def _get_own_repr(self):
    value = self._value
    try:
        return '%d: %s' % (value, reverse_mapping[value])
    except KeyError:
        return str(value)