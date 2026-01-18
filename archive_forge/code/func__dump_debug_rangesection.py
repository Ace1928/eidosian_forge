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
def _dump_debug_rangesection(self, di, range_lists_sec):
    ver5 = range_lists_sec.version >= 5
    section_name = (di.debug_rnglists_sec if ver5 else di.debug_ranges_sec).name
    addr_size = di.config.default_address_size
    addr_width = addr_size * 2
    line_template = '    %%08x %%0%dx %%0%dx %%s' % (addr_width, addr_width)
    base_template = '    %%08x %%0%dx (base address)' % addr_width
    base_template_indexed = '    %%08x %%0%dx (base address index) %%0%dx (base address)' % (addr_width, addr_width)
    cu_map = {die.attributes['DW_AT_ranges'].value: cu for cu in di.iter_CUs() for die in cu.iter_DIEs() if 'DW_AT_ranges' in die.attributes}
    rcus = list(range_lists_sec.iter_CUs()) if ver5 else None
    rcu_index = 0
    next_rcu_offset = 0
    range_lists = list(range_lists_sec.iter_range_lists())
    if len(range_lists) == 0:
        self._emitline("\nSection '%s' has no debugging data." % section_name)
        return
    self._emitline('Contents of the %s section:\n\n\n' % section_name)
    if not ver5:
        self._emitline('    Offset   Begin    End')
    for range_list in range_lists:
        if ver5 and range_list[0].entry_offset > next_rcu_offset:
            while range_list[0].entry_offset > next_rcu_offset:
                rcu = rcus[rcu_index]
                self._dump_debug_rnglists_CU_header(rcu)
                next_rcu_offset = rcu.offset_after_length + rcu.unit_length
                rcu_index += 1
            self._emitline('    Offset   Begin    End')
        self._dump_rangelist(range_list, cu_map, ver5, line_template, base_template, base_template_indexed, range_lists_sec)