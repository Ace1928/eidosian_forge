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
class WriteWithCrc(io.RawIOBase):
    """Thin wrapper for file object to calculate crc32 when write called."""

    def __init__(self, fp: BinaryIO):
        self._fp = fp
        self.digest = 0

    def write(self, data):
        self.digest = calculate_crc32(data, self.digest)
        return self._fp.write(data)

    def tell(self):
        return self._fp.tell()