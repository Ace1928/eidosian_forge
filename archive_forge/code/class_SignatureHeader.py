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
class SignatureHeader:
    """The SignatureHeader class hold information of a signature header of archive."""
    __slots__ = ['version', 'startheadercrc', 'nextheaderofs', 'nextheadersize', 'nextheadercrc']

    def __init__(self) -> None:
        self.version = (P7ZIP_MAJOR_VERSION, P7ZIP_MINOR_VERSION)
        self.startheadercrc: int = -1
        self.nextheaderofs: int = -1
        self.nextheadersize: int = -1
        self.nextheadercrc: int = -1

    @classmethod
    def retrieve(cls, file: BinaryIO):
        obj = cls()
        obj._read(file)
        return obj

    def _read(self, file: BinaryIO) -> None:
        file.seek(len(MAGIC_7Z), 0)
        major_version = file.read(1)
        minor_version = file.read(1)
        self.version = (major_version, minor_version)
        self.startheadercrc, _ = read_uint32(file)
        self.nextheaderofs, data = read_real_uint64(file)
        crc = calculate_crc32(data)
        self.nextheadersize, data = read_real_uint64(file)
        crc = calculate_crc32(data, crc)
        self.nextheadercrc, data = read_uint32(file)
        crc = calculate_crc32(data, crc)
        if crc != self.startheadercrc:
            raise Bad7zFile('invalid header data')

    def calccrc(self, length: int, header_crc: int):
        self.nextheadersize = length
        self.nextheadercrc = header_crc
        buf = io.BytesIO()
        write_real_uint64(buf, self.nextheaderofs)
        write_real_uint64(buf, self.nextheadersize)
        write_uint32(buf, self.nextheadercrc)
        startdata = buf.getvalue()
        self.startheadercrc = calculate_crc32(startdata)

    def write(self, file: BinaryIO):
        assert self.startheadercrc >= 0
        assert self.nextheadercrc >= 0
        assert self.nextheaderofs >= 0
        assert self.nextheadersize > 0
        file.seek(0, 0)
        write_bytes(file, MAGIC_7Z)
        write_byte(file, self.version[0])
        write_byte(file, self.version[1])
        write_uint32(file, self.startheadercrc)
        write_real_uint64(file, self.nextheaderofs)
        write_real_uint64(file, self.nextheadersize)
        write_uint32(file, self.nextheadercrc)

    def _write_skelton(self, file: BinaryIO):
        file.seek(0, 0)
        write_bytes(file, MAGIC_7Z)
        write_byte(file, self.version[0])
        write_byte(file, self.version[1])
        write_uint32(file, 1)
        write_real_uint64(file, 2)
        write_real_uint64(file, 3)
        write_uint32(file, 4)