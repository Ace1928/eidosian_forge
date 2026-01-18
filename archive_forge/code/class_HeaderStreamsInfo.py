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
class HeaderStreamsInfo(StreamsInfo):
    """
    Header version of StreamsInfo
    """

    def __init__(self):
        super().__init__()
        self.packinfo: PackInfo = PackInfo()
        self.unpackinfo: UnpackInfo = UnpackInfo()
        self.unpackinfo.numfolders = 1

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        write_byte(file, PROPERTY.ENCODED_HEADER)
        self.packinfo.write(file)
        self.unpackinfo.write(file)
        write_byte(file, PROPERTY.END)