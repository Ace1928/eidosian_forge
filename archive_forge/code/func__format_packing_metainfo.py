from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _format_packing_metainfo(self):
    """The struct packing format used for the items' packing formats."""
    return '8s' * self._list_len