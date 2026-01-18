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
class UnpackInfo:
    """combines multiple folders"""
    __slots__ = ['numfolders', 'folders', 'datastreamidx']

    @classmethod
    def retrieve(cls, file: BinaryIO):
        obj = cls()
        obj._read(file)
        return obj

    def __init__(self):
        self.numfolders: int = 0
        self.folders = []
        self.datastreamidx = None

    def _read(self, file: BinaryIO):
        pid = file.read(1)
        if pid != PROPERTY.FOLDER:
            raise Bad7zFile('folder id expected but %s found' % repr(pid))
        self.numfolders = read_uint64(file)
        self.folders = []
        external = read_byte(file)
        if external == 0:
            self.folders = [Folder.retrieve(file) for _ in range(self.numfolders)]
        else:
            datastreamidx = read_uint64(file)
            current_pos = file.tell()
            file.seek(datastreamidx, 0)
            self.folders = [Folder.retrieve(file) for _ in range(self.numfolders)]
            file.seek(current_pos, 0)
        self._retrieve_coders_info(file)

    def _retrieve_coders_info(self, file: BinaryIO):
        pid = file.read(1)
        if pid != PROPERTY.CODERS_UNPACK_SIZE:
            raise Bad7zFile('coders unpack size id expected but %s found' % repr(pid))
        for folder in self.folders:
            for c in folder.coders:
                for _ in range(c['numoutstreams']):
                    folder.unpacksizes.append(read_uint64(file))
        pid = file.read(1)
        if pid == PROPERTY.CRC:
            defined = read_boolean(file, self.numfolders, checkall=True)
            crcs = read_crcs(file, self.numfolders)
            for idx, folder in enumerate(self.folders):
                folder.digestdefined = defined[idx]
                folder.crc = crcs[idx]
            pid = file.read(1)
        if pid != PROPERTY.END:
            raise Bad7zFile('end id expected but 0x{:02x} found at 0x{:08x}'.format(ord(pid), file.tell()))

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        assert self.numfolders == len(self.folders)
        file.write(PROPERTY.UNPACK_INFO)
        file.write(PROPERTY.FOLDER)
        write_uint64(file, self.numfolders)
        write_byte(file, b'\x00')
        for folder in self.folders:
            folder.write(file)
        write_byte(file, PROPERTY.CODERS_UNPACK_SIZE)
        for folder in self.folders:
            for s in folder.unpacksizes:
                write_uint64(file, s)
        write_byte(file, PROPERTY.END)