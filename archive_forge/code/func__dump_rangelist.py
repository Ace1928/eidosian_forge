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
def _dump_rangelist(self, range_list, cu_map, ver5, line_template, base_template, base_template_indexed, range_lists_sec):
    first = range_list[0]
    base_ip = _get_cu_base(cu_map[first.entry_offset])
    raw_v5_rangelist = None
    for entry in range_list:
        if isinstance(entry, RangeEntry):
            postfix = ' (start == end)' if entry.begin_offset == entry.end_offset else ''
            self._emitline(line_template % (entry.entry_offset if ver5 else first.entry_offset, (0 if entry.is_absolute else base_ip) + entry.begin_offset, (0 if entry.is_absolute else base_ip) + entry.end_offset, postfix))
        elif isinstance(entry, RangeBaseAddressEntry):
            base_ip = entry.base_address
            raw_v5_entry = None
            if ver5:
                if not raw_v5_rangelist:
                    raw_v5_rangelist = range_lists_sec.get_range_list_at_offset_ex(range_list[0].entry_offset)
                raw_v5_entry = next((re for re in raw_v5_rangelist if re.entry_offset == entry.entry_offset))
            if raw_v5_entry and raw_v5_entry.entry_type == 'DW_RLE_base_addressx':
                self._emitline(base_template_indexed % (entry.entry_offset, raw_v5_entry.index, entry.base_address))
            else:
                self._emitline(base_template % (entry.entry_offset if ver5 else first.entry_offset, entry.base_address))
        else:
            raise NotImplementedError('Unknown object in a range list')
    last = range_list[-1]
    self._emitline('    %08x <End of list>' % (last.entry_offset + last.entry_length if ver5 else first.entry_offset))