import collections
import functools
import os
import re
import struct
import sys
import warnings
from typing import IO, Dict, Iterator, NamedTuple, Optional, Tuple
class _ELFFileHeader:

    class _InvalidELFFileHeader(ValueError):
        """
        An invalid ELF file header was found.
        """
    ELF_MAGIC_NUMBER = 2135247942
    ELFCLASS32 = 1
    ELFCLASS64 = 2
    ELFDATA2LSB = 1
    ELFDATA2MSB = 2
    EM_386 = 3
    EM_S390 = 22
    EM_ARM = 40
    EM_X86_64 = 62
    EF_ARM_ABIMASK = 4278190080
    EF_ARM_ABI_VER5 = 83886080
    EF_ARM_ABI_FLOAT_HARD = 1024

    def __init__(self, file: IO[bytes]) -> None:

        def unpack(fmt: str) -> int:
            try:
                data = file.read(struct.calcsize(fmt))
                result: Tuple[int, ...] = struct.unpack(fmt, data)
            except struct.error:
                raise _ELFFileHeader._InvalidELFFileHeader()
            return result[0]
        self.e_ident_magic = unpack('>I')
        if self.e_ident_magic != self.ELF_MAGIC_NUMBER:
            raise _ELFFileHeader._InvalidELFFileHeader()
        self.e_ident_class = unpack('B')
        if self.e_ident_class not in {self.ELFCLASS32, self.ELFCLASS64}:
            raise _ELFFileHeader._InvalidELFFileHeader()
        self.e_ident_data = unpack('B')
        if self.e_ident_data not in {self.ELFDATA2LSB, self.ELFDATA2MSB}:
            raise _ELFFileHeader._InvalidELFFileHeader()
        self.e_ident_version = unpack('B')
        self.e_ident_osabi = unpack('B')
        self.e_ident_abiversion = unpack('B')
        self.e_ident_pad = file.read(7)
        format_h = '<H' if self.e_ident_data == self.ELFDATA2LSB else '>H'
        format_i = '<I' if self.e_ident_data == self.ELFDATA2LSB else '>I'
        format_q = '<Q' if self.e_ident_data == self.ELFDATA2LSB else '>Q'
        format_p = format_i if self.e_ident_class == self.ELFCLASS32 else format_q
        self.e_type = unpack(format_h)
        self.e_machine = unpack(format_h)
        self.e_version = unpack(format_i)
        self.e_entry = unpack(format_p)
        self.e_phoff = unpack(format_p)
        self.e_shoff = unpack(format_p)
        self.e_flags = unpack(format_i)
        self.e_ehsize = unpack(format_h)
        self.e_phentsize = unpack(format_h)
        self.e_phnum = unpack(format_h)
        self.e_shentsize = unpack(format_h)
        self.e_shnum = unpack(format_h)
        self.e_shstrndx = unpack(format_h)