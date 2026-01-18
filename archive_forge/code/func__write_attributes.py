import functools
import io
import operator
import os
import struct
from binascii import unhexlify
from functools import reduce
from io import BytesIO
from operator import and_, or_
from struct import pack, unpack
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from py7zr.compressor import SevenZipCompressor, SevenZipDecompressor
from py7zr.exceptions import Bad7zFile
from py7zr.helpers import ArchiveTimestamp, calculate_crc32
from py7zr.properties import DEFAULT_FILTERS, MAGIC_7Z, PROPERTY
def _write_attributes(self, file):
    defined = []
    num_defined = 0
    for f in self.files:
        if 'attributes' in f.keys() and f['attributes'] is not None:
            defined.append(True)
            num_defined += 1
        else:
            defined.append(False)
    size = num_defined * 4 + 2
    if num_defined != len(defined):
        size += bits_to_bytes(num_defined)
    write_byte(file, PROPERTY.ATTRIBUTES)
    write_uint64(file, size)
    write_boolean(file, defined, all_defined=True)
    write_byte(file, b'\x00')
    for i, f in enumerate(self.files):
        if defined[i]:
            write_uint32(file, f['attributes'])