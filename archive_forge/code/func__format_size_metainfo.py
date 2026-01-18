from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _format_size_metainfo(self):
    """The struct packing format used for the items' storage offsets."""
    return 'q' * (self._list_len + 1)