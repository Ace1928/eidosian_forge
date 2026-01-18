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
def _read_dwarf_section(self, section, relocate_dwarf_sections):
    """ Read the contents of a DWARF section from the stream and return a
            DebugSectionDescriptor. Apply relocations if asked to.
        """
    phantom_bytes = self.has_phantom_bytes()
    section_stream = BytesIO()
    section_data = section.data()
    section_stream.write(section_data[::2] if phantom_bytes else section_data)
    if relocate_dwarf_sections:
        reloc_handler = RelocationHandler(self)
        reloc_section = reloc_handler.find_relocations_for_section(section)
        if reloc_section is not None:
            if phantom_bytes:
                raise ELFParseError('This binary has relocations in the DWARF sections, currently not supported.')
            else:
                reloc_handler.apply_section_relocations(section_stream, reloc_section)
    return DebugSectionDescriptor(stream=section_stream, name=section.name, global_offset=section['sh_offset'], size=section.data_size // 2 if phantom_bytes else section.data_size, address=section['sh_addr'])