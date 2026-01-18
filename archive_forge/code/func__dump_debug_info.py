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
def _dump_debug_info(self):
    """ Dump the debugging info section.
        """
    if not self._dwarfinfo.has_debug_info:
        return
    self._emitline('Contents of the %s section:\n' % self._dwarfinfo.debug_info_sec.name)
    section_offset = self._dwarfinfo.debug_info_sec.global_offset
    for cu in self._dwarfinfo.iter_CUs():
        self._emitline('  Compilation Unit @ offset %s:' % self._format_hex(cu.cu_offset, alternate=True))
        self._emitline('   Length:        %s (%s)' % (self._format_hex(cu['unit_length']), '%s-bit' % cu.dwarf_format()))
        self._emitline('   Version:       %s' % cu['version'])
        if cu['version'] >= 5:
            if cu.header.get('unit_type', ''):
                unit_type = cu.header.unit_type
                self._emitline('   Unit Type:     %s (%d)' % (unit_type, ENUM_DW_UT.get(cu.header.unit_type, 0)))
                self._emitline('   Abbrev Offset: %s' % self._format_hex(cu['debug_abbrev_offset'], alternate=True))
                self._emitline('   Pointer Size:  %s' % cu['address_size'])
                if unit_type in ('DW_UT_skeleton', 'DW_UT_split_compile'):
                    self._emitline('   Dwo id:        %s' % cu['dwo_id'])
                elif unit_type in ('DW_UT_type', 'DW_UT_split_type'):
                    self._emitline('   Signature:     0x%x' % cu['type_signature'])
                    self._emitline('   Type Offset:   0x%x' % cu['type_offset'])
        else:
            (self._emitline('   Abbrev Offset: %s' % self._format_hex(cu['debug_abbrev_offset'], alternate=True)),)
            self._emitline('   Pointer Size:  %s' % cu['address_size'])
        die_depth = 0
        current_function = None
        for die in cu.iter_DIEs():
            if die.tag == 'DW_TAG_subprogram':
                current_function = die
            self._emitline(' <%s><%x>: Abbrev Number: %s%s' % (die_depth, die.offset, die.abbrev_code, ' (%s)' % die.tag if not die.is_null() else ''))
            if die.is_null():
                die_depth -= 1
                continue
            for attr in die.attributes.values():
                name = attr.name
                if isinstance(name, int):
                    name = 'Unknown AT value: %x' % name
                attr_desc = describe_attr_value(attr, die, section_offset)
                if 'DW_OP_fbreg' in attr_desc and current_function and (not 'DW_AT_frame_base' in current_function.attributes):
                    postfix = ' [without dw_at_frame_base]'
                else:
                    postfix = ''
                self._emitline('    <%x>   %-18s: %s%s' % (attr.offset, name, attr_desc, postfix))
            if die.has_children:
                die_depth += 1
    self._emitline()