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
def _write_times(self, fp: Union[BinaryIO, WriteWithCrc], propid, name: str) -> None:
    write_byte(fp, propid)
    defined = []
    num_defined = 0
    for f in self.files:
        if name in f.keys():
            if f[name] is not None:
                defined.append(True)
                num_defined += 1
                continue
        defined.append(False)
    size = num_defined * 8 + 2
    if not reduce(and_, defined, True):
        size += bits_to_bytes(num_defined)
    write_uint64(fp, size)
    write_boolean(fp, defined, all_defined=True)
    write_byte(fp, b'\x00')
    for i, file in enumerate(self.files):
        if defined[i]:
            write_real_uint64(fp, file[name])
        else:
            pass