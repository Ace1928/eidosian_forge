from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
def _to_file(self, f):
    """Private friend method called by encoder to write a blob to a file."""
    if self.allocated_size <= 250 and self.compression == 0:
        f.write(spack('<B', self.allocated_size))
        f.write(spack('<B', self.used_size))
        f.write(lencode(self.data_size))
    else:
        f.write(spack('<BQ', 253, self.allocated_size))
        f.write(spack('<BQ', 253, self.used_size))
        f.write(spack('<BQ', 253, self.data_size))
    f.write(spack('B', self.compression))
    if self.use_checksum:
        f.write(b'\xff' + hashlib.md5(self.compressed).digest())
    else:
        f.write(b'\x00')
    if self.compression == 0:
        alignment = 8 - (f.tell() + 1) % 8
        f.write(spack('<B', alignment))
        f.write(b'\x00' * alignment)
    else:
        f.write(spack('<B', 0))
    f.write(self.compressed)
    f.write(b'\x00' * (self.allocated_size - self.used_size))