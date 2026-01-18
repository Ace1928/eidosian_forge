import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def _unpack_stat(packed_stat):
    """Turn a packed_stat back into the stat fields.

    This is meant as a debugging tool, should not be used in real code.
    """
    st_size, st_mtime, st_ctime, st_dev, st_ino, st_mode = struct.unpack('>6L', binascii.a2b_base64(packed_stat))
    return dict(st_size=st_size, st_mtime=st_mtime, st_ctime=st_ctime, st_dev=st_dev, st_ino=st_ino, st_mode=st_mode)