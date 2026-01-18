from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@staticmethod
def _extract_recreation_code(value):
    """Used in concert with _back_transforms_mapping to convert values
        into the appropriate Python objects when retrieving them from
        the list as well as when storing them."""
    if not isinstance(value, (str, bytes, None.__class__)):
        return 0
    elif isinstance(value, str):
        return 1
    elif isinstance(value, bytes):
        return 2
    else:
        return 3