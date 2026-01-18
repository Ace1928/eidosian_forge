import os
from . import BioSeq
from . import Loader
from . import DBUtils
@staticmethod
def _bytearray_to_str(s):
    """If s is bytes or bytearray, convert to a string (PRIVATE)."""
    if isinstance(s, (bytes, bytearray)):
        return s.decode()
    return s