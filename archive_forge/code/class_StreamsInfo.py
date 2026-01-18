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
class StreamsInfo:
    """information about compressed streams"""
    __slots__ = ['packinfo', 'unpackinfo', 'substreamsinfo']

    def __init__(self):
        self.packinfo: Optional[PackInfo] = None
        self.unpackinfo: Optional[UnpackInfo] = None
        self.substreamsinfo: Optional[SubstreamsInfo] = None

    @classmethod
    def retrieve(cls, file: BinaryIO):
        obj = cls()
        obj.read(file)
        return obj

    def read(self, file: BinaryIO) -> None:
        pid = file.read(1)
        if pid == PROPERTY.PACK_INFO:
            self.packinfo = PackInfo.retrieve(file)
            pid = file.read(1)
        if pid == PROPERTY.UNPACK_INFO:
            self.unpackinfo = UnpackInfo.retrieve(file)
            pid = file.read(1)
        if pid == PROPERTY.SUBSTREAMS_INFO:
            if self.unpackinfo is None:
                raise Bad7zFile('Header is broken')
            self.substreamsinfo = SubstreamsInfo.retrieve(file, self.unpackinfo.numfolders, self.unpackinfo.folders)
            pid = file.read(1)
        if pid != PROPERTY.END:
            raise Bad7zFile('end id expected but %s found' % repr(pid))

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        write_byte(file, PROPERTY.MAIN_STREAMS_INFO)
        if self.packinfo is not None:
            self.packinfo.write(file)
        if self.unpackinfo is not None:
            self.unpackinfo.write(file)
        if self.substreamsinfo is not None:
            self.substreamsinfo.write(file)
        write_byte(file, PROPERTY.END)