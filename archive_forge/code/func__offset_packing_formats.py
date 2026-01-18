from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
@property
def _offset_packing_formats(self):
    return self._offset_data_start + self._allocated_offsets[-1]