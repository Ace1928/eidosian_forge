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
def _get_section_name(self, section_header):
    """ Given a section header, find this section's name in the file's
            string table
        """
    if self._section_header_stringtable is None:
        raise ELFParseError('String Table not found')
    name_offset = section_header['sh_name']
    return self._section_header_stringtable.get_string(name_offset)