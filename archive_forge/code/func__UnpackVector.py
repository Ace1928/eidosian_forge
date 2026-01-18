import array
import contextlib
import enum
import struct
def _UnpackVector(fmt, buf, length):
    byte_width = len(buf) // length
    return struct.unpack('<%d%s' % (length, fmt[byte_width]), buf)