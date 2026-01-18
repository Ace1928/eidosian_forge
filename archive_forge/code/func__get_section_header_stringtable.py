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
def _get_section_header_stringtable(self):
    """ Get the string table section corresponding to the section header
            table.
        """
    stringtable_section_num = self.get_shstrndx()
    stringtable_section_header = self._get_section_header(stringtable_section_num)
    if stringtable_section_header is None:
        return None
    return StringTableSection(header=stringtable_section_header, name='', elffile=self)