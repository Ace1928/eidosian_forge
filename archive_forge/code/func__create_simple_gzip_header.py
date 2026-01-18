import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _create_simple_gzip_header(compresslevel: int, mtime=None) -> bytes:
    """
    Write a simple gzip header with no extra fields.
    :param compresslevel: Compresslevel used to determine the xfl bytes.
    :param mtime: The mtime (must support conversion to a 32-bit integer).
    :return: A bytes object representing the gzip header.
    """
    if mtime is None:
        mtime = time.time()
    if compresslevel == _COMPRESS_LEVEL_BEST:
        xfl = 2
    elif compresslevel == _COMPRESS_LEVEL_FAST:
        xfl = 4
    else:
        xfl = 0
    return struct.pack('<BBBBLBB', 31, 139, 8, 0, int(mtime), xfl, 255)