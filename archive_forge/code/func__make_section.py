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
def _make_section(self, section_header):
    """ Create a section object of the appropriate type
        """
    name = self._get_section_name(section_header)
    sectype = section_header['sh_type']
    if sectype == 'SHT_STRTAB':
        return StringTableSection(section_header, name, self)
    elif sectype == 'SHT_NULL':
        return NullSection(section_header, name, self)
    elif sectype in ('SHT_SYMTAB', 'SHT_DYNSYM', 'SHT_SUNW_LDYNSYM'):
        return self._make_symbol_table_section(section_header, name)
    elif sectype == 'SHT_SYMTAB_SHNDX':
        return self._make_symbol_table_index_section(section_header, name)
    elif sectype == 'SHT_SUNW_syminfo':
        return self._make_sunwsyminfo_table_section(section_header, name)
    elif sectype == 'SHT_GNU_verneed':
        return self._make_gnu_verneed_section(section_header, name)
    elif sectype == 'SHT_GNU_verdef':
        return self._make_gnu_verdef_section(section_header, name)
    elif sectype == 'SHT_GNU_versym':
        return self._make_gnu_versym_section(section_header, name)
    elif sectype in ('SHT_REL', 'SHT_RELA'):
        return RelocationSection(section_header, name, self)
    elif sectype == 'SHT_DYNAMIC':
        return DynamicSection(section_header, name, self)
    elif sectype == 'SHT_NOTE':
        return NoteSection(section_header, name, self)
    elif sectype == 'SHT_PROGBITS' and name == '.stab':
        return StabSection(section_header, name, self)
    elif sectype == 'SHT_ARM_ATTRIBUTES':
        return ARMAttributesSection(section_header, name, self)
    elif sectype == 'SHT_RISCV_ATTRIBUTES':
        return RISCVAttributesSection(section_header, name, self)
    elif sectype == 'SHT_HASH':
        return self._make_elf_hash_section(section_header, name)
    elif sectype == 'SHT_GNU_HASH':
        return self._make_gnu_hash_section(section_header, name)
    elif sectype == 'SHT_RELR':
        return RelrRelocationSection(section_header, name, self)
    else:
        return Section(section_header, name, self)