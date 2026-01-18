import io
from io import BytesIO
import os
import struct
import zlib
from ..common.exceptions import ELFError, ELFParseError
from ..common.utils import struct_parse, elf_assert
from .structs import ELFStructs
from .sections import (
from .dynamic import DynamicSection, DynamicSegment
from .relocation import (RelocationSection, RelocationHandler,
from .gnuversions import (
from .segments import Segment, InterpSegment, NoteSegment
from ..dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig
from ..ehabi.ehabiinfo import EHABIInfo
from .hash import ELFHashSection, GNUHashSection
from .constants import SHN_INDICES
@staticmethod
def _decompress_dwarf_section(section):
    """ Returns the uncompressed contents of the provided DWARF section.
        """
    assert section.size > 12, 'Unsupported compression format.'
    section.stream.seek(0)
    compression_type = section.stream.read(4)
    assert compression_type == b'ZLIB', 'Invalid compression type: %r' % compression_type
    uncompressed_size = struct.unpack('>Q', section.stream.read(8))[0]
    decompressor = zlib.decompressobj()
    uncompressed_stream = BytesIO()
    while True:
        chunk = section.stream.read(PAGESIZE)
        if not chunk:
            break
        uncompressed_stream.write(decompressor.decompress(chunk))
    uncompressed_stream.write(decompressor.flush())
    uncompressed_stream.seek(0, io.SEEK_END)
    size = uncompressed_stream.tell()
    assert uncompressed_size == size, 'Wrong uncompressed size: expected %r, but got %r' % (uncompressed_size, size)
    return section._replace(stream=uncompressed_stream, size=size)