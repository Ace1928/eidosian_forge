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
def _dump_debug_aranges(self):
    """ Dump the aranges table
        """
    aranges_table = self._dwarfinfo.get_aranges()
    if aranges_table == None:
        return
    unordered_entries = aranges_table._get_entries(need_empty=True)
    if len(unordered_entries) == 0:
        self._emitline()
        self._emitline("Section '.debug_aranges' has no debugging data.")
        return
    self._emitline('Contents of the %s section:' % self._dwarfinfo.debug_aranges_sec.name)
    self._emitline()
    prev_offset = None
    for entry in unordered_entries:
        if prev_offset != entry.info_offset:
            if entry != unordered_entries[0]:
                self._emitline('    %s %s' % (self._format_hex(0, fullhex=True, lead0x=False), self._format_hex(0, fullhex=True, lead0x=False)))
            self._emitline('  Length:                   %d' % entry.unit_length)
            self._emitline('  Version:                  %d' % entry.version)
            self._emitline('  Offset into .debug_info:  0x%x' % entry.info_offset)
            self._emitline('  Pointer Size:             %d' % entry.address_size)
            self._emitline('  Segment Size:             %d' % entry.segment_size)
            self._emitline()
            self._emitline('    Address            Length')
        if entry.begin_addr != 0 or entry.length != 0:
            self._emitline('    %s %s' % (self._format_hex(entry.begin_addr, fullhex=True, lead0x=False), self._format_hex(entry.length, fullhex=True, lead0x=False)))
        prev_offset = entry.info_offset
    self._emitline('    %s %s' % (self._format_hex(0, fullhex=True, lead0x=False), self._format_hex(0, fullhex=True, lead0x=False)))