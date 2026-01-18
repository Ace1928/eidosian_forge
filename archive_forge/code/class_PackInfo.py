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
class PackInfo:
    """information about packed streams"""
    __slots__ = ['packpos', 'numstreams', 'packsizes', 'packpositions', 'crcs', 'digestdefined', 'enable_digests']

    def __init__(self) -> None:
        self.packpos: int = 0
        self.numstreams: int = 0
        self.packsizes: List[int] = []
        self.digestdefined: List[bool] = []
        self.crcs: List[int] = []
        self.enable_digests: bool = True

    @classmethod
    def retrieve(cls, file: BinaryIO):
        return cls()._read(file)

    def _read(self, file: BinaryIO):
        self.packpos = read_uint64(file)
        self.numstreams = read_uint64(file)
        pid = file.read(1)
        if pid == PROPERTY.SIZE:
            self.packsizes = [read_uint64(file) for _ in range(self.numstreams)]
            pid = file.read(1)
            if pid == PROPERTY.CRC:
                self.digestdefined = read_boolean(file, self.numstreams, True)
                for crcexist in self.digestdefined:
                    self.enable_digests = True
                    if crcexist:
                        self.crcs.append(read_uint32(file)[0])
                pid = file.read(1)
        if pid != PROPERTY.END:
            raise Bad7zFile('end id expected but %s found' % repr(pid))
        self.packpositions = [sum(self.packsizes[:i]) for i in range(self.numstreams + 1)]
        self.enable_digests = len(self.crcs) > 0
        return self

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        assert self.packpos is not None
        assert self.numstreams == len(self.packsizes)
        write_byte(file, PROPERTY.PACK_INFO)
        write_uint64(file, self.packpos)
        write_uint64(file, self.numstreams)
        write_byte(file, PROPERTY.SIZE)
        for size in self.packsizes:
            write_uint64(file, size)
        self.enable_digests = functools.reduce(operator.or_, self.digestdefined, self.enable_digests)
        if self.enable_digests:
            assert len(self.crcs) == self.numstreams
            write_byte(file, PROPERTY.CRC)
            write_boolean(file, self.digestdefined, True)
            for i in range(self.numstreams):
                if self.digestdefined[i]:
                    write_uint32(file, self.crcs[i])
        write_byte(file, PROPERTY.END)