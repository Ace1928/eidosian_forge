import copy
import weakref
from pyomo.common.autoslots import AutoSlots
def _convert_descend_into(value):
    """Converts the descend_into keyword to a function"""
    if hasattr(value, '__call__'):
        return value
    elif value:
        return _convert_descend_into._true
    else:
        return _convert_descend_into._false