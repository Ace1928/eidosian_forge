import re
from numba.core import types
def _fix_lead_digit(text):
    """
    Fix text with leading digit
    """
    if text and text[0].isdigit():
        return '_' + text
    else:
        return text