from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
def _get_back_transform(self, position):
    """Gets the back transformation function for a single value."""
    if position >= self._list_len or self._list_len < 0:
        raise IndexError('Requested position out of range.')
    transform_code = struct.unpack_from('b', self.shm.buf, self._offset_back_transform_codes + position)[0]
    transform_function = self._back_transforms_mapping[transform_code]
    return transform_function