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
def _dump_frames_interp_info(self, section, cfi_entries):
    """ Dump interpreted (decoded) frame information in a section.

        `section` is the Section instance that contains the call frame info
        while `cfi_entries` must be an iterable that yields the sequence of
        CIE or FDE instances.
        """
    self._emitline('Contents of the %s section:' % section.name)
    for entry in cfi_entries:
        if isinstance(entry, CIE):
            self._emitline('\n%08x %s %s CIE "%s" cf=%d df=%d ra=%d' % (entry.offset, self._format_hex(entry['length'], fullhex=True, lead0x=False), self._format_hex(entry['CIE_id'], fieldsize=8, lead0x=False), bytes2str(entry['augmentation']), entry['code_alignment_factor'], entry['data_alignment_factor'], entry['return_address_register']))
            ra_regnum = entry['return_address_register']
        elif isinstance(entry, FDE):
            self._emitline('\n%08x %s %s FDE cie=%08x pc=%s..%s' % (entry.offset, self._format_hex(entry['length'], fullhex=True, lead0x=False), self._format_hex(entry['CIE_pointer'], fieldsize=8, lead0x=False), entry.cie.offset, self._format_hex(entry['initial_location'], fullhex=True, lead0x=False), self._format_hex(entry['initial_location'] + entry['address_range'], fullhex=True, lead0x=False)))
            ra_regnum = entry.cie['return_address_register']
            if len(entry.get_decoded().table) == len(entry.cie.get_decoded().table):
                continue
        else:
            assert isinstance(entry, ZERO)
            self._emitline('\n%08x ZERO terminator' % entry.offset)
            continue
        decoded_table = entry.get_decoded()
        if len(decoded_table.table) == 0:
            continue
        self._emit('   LOC')
        self._emit('  ' if entry.structs.address_size == 4 else '          ')
        self._emit(' CFA      ')
        decoded_table = entry.get_decoded()
        reg_order = sorted(decoded_table.reg_order)
        if len(decoded_table.reg_order):
            for regnum in reg_order:
                if regnum == ra_regnum:
                    self._emit('ra      ')
                    continue
                self._emit('%-6s' % describe_reg_name(regnum))
        self._emitline()
        for line in decoded_table.table:
            self._emit(self._format_hex(line['pc'], fullhex=True, lead0x=False))
            if line['cfa'] is not None:
                s = describe_CFI_CFA_rule(line['cfa'])
            else:
                s = 'u'
            self._emit(' %-9s' % s)
            for regnum in reg_order:
                if regnum in line:
                    s = describe_CFI_register_rule(line[regnum])
                else:
                    s = 'u'
                self._emit('%-6s' % s)
            self._emitline()
    self._emitline()