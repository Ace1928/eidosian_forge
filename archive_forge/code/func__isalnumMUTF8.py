from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return c.isalnum() or len(c.encode('utf-8')) > 1