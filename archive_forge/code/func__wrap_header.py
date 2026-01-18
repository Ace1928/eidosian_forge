import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def _wrap_header(header, version):
    """
    Takes a stringified header, and attaches the prefix and padding to it
    """
    import struct
    assert version is not None
    fmt, encoding = _header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = ARRAY_ALIGN - (MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN
    try:
        header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    except struct.error:
        msg = 'Header length {} too big for version={}'.format(hlen, version)
        raise ValueError(msg) from None
    return header_prefix + header + b' ' * padlen + b'\n'