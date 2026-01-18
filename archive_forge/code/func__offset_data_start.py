from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _offset_data_start(self):
    return (self._list_len + 2) * 8