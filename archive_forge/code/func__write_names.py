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
def _write_names(self, file: Union[BinaryIO, WriteWithCrc]):
    name_defined = 0
    names = []
    name_size = 0
    for f in self.files:
        if f.get('filename', None) is not None:
            name_defined += 1
            names.append(f['filename'])
            name_size += len(f['filename'].encode('utf-16LE')) + 2
    if name_defined > 0:
        write_byte(file, PROPERTY.NAME)
        write_uint64(file, name_size + 1)
        write_byte(file, b'\x00')
        for n in names:
            write_utf16(file, n)