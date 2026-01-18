import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _handle_pad_byte(fid, size):
    if size % 2:
        fid.seek(1, 1)