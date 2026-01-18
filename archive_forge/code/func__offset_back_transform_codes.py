from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _offset_back_transform_codes(self):
    return self._offset_packing_formats + self._list_len * 8