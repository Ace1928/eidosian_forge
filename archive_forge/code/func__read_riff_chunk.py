import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _read_riff_chunk(fid):
    str1 = fid.read(4)
    if str1 == b'RIFF':
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_big_endian = True
        fmt = '>I'
    else:
        raise ValueError(f"File format {repr(str1)} not understood. Only 'RIFF' and 'RIFX' supported.")
    file_size = struct.unpack(fmt, fid.read(4))[0] + 8
    str2 = fid.read(4)
    if str2 != b'WAVE':
        raise ValueError(f'Not a WAV file. RIFF form type is {repr(str2)}.')
    return (file_size, is_big_endian)