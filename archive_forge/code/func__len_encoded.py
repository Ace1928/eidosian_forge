import re
from numba.core import types
def _len_encoded(string):
    """
    Prefix string with digit indicating the length.
    Add underscore if string is prefixed with digits.
    """
    string = _fix_lead_digit(string)
    return '%u%s' % (len(string), string)