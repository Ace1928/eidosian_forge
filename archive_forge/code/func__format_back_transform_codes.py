from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _format_back_transform_codes(self):
    """The struct packing format used for the items' back transforms."""
    return 'b' * self._list_len