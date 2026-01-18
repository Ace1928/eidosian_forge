import argparse
import os, sys
import re
import string
import traceback
import itertools
from elftools import __version__
from elftools.common.exceptions import ELFError
from elftools.common.utils import bytes2str, iterbytes
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.enums import ENUM_D_TAG
from elftools.elf.segments import InterpSegment
from elftools.elf.sections import (
from elftools.elf.gnuversions import (
from elftools.elf.relocation import RelocationSection
from elftools.elf.descriptions import (
from elftools.elf.constants import E_FLAGS
from elftools.elf.constants import E_FLAGS_MASKS
from elftools.elf.constants import SH_FLAGS
from elftools.elf.constants import SHN_INDICES
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.descriptions import (
from elftools.dwarf.constants import (
from elftools.dwarf.locationlists import LocationParser, LocationEntry, LocationViewPair, BaseAddressEntry as LocBaseAddressEntry, LocationListsPair
from elftools.dwarf.ranges import RangeEntry, BaseAddressEntry as RangeBaseAddressEntry, RangeListsPair
from elftools.dwarf.callframe import CIE, FDE, ZERO
from elftools.ehabi.ehabiinfo import CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry
from elftools.dwarf.enums import ENUM_DW_UT
def _get_symbol_shndx(self, symbol, symbol_index, symtab_index):
    """ Get the index into the section header table for the "symbol"
            at "symbol_index" located in the symbol table with section index
            "symtab_index".
        """
    symbol_shndx = symbol['st_shndx']
    if symbol_shndx != SHN_INDICES.SHN_XINDEX:
        return symbol_shndx
    if self._shndx_sections is None:
        self._shndx_sections = {sec.symboltable: sec for sec in self.elffile.iter_sections() if isinstance(sec, SymbolTableIndexSection)}
    return self._shndx_sections[symtab_index].get_section_index(symbol_index)