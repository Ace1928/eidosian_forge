from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
def _make_filename():
    """Create a random filename for the shared memory object."""
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name