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
def _dump_loclist(self, loc_list, line_template, cu_map):
    in_views = False
    has_views = False
    base_ip = None
    loc_entry_count = 0
    cu = None
    for entry in loc_list:
        if isinstance(entry, LocationViewPair):
            has_views = in_views = True
            self._emitline('    %08x v%015x v%015x location view pair' % (entry.entry_offset, entry.begin, entry.end))
        else:
            if in_views:
                in_views = False
                self._emitline('')
            if cu_map is None:
                base_ip = 0
            elif cu is None:
                cu = cu_map.get(entry.entry_offset, False)
                if not cu:
                    raise ValueError("Location list can't be tracked to a CU")
            if isinstance(entry, LocationEntry):
                if base_ip is None and (not entry.is_absolute):
                    base_ip = _get_cu_base(cu)
                begin_offset = (0 if entry.is_absolute else base_ip) + entry.begin_offset
                end_offset = (0 if entry.is_absolute else base_ip) + entry.end_offset
                expr = describe_DWARF_expr(entry.loc_expr, cu.structs, cu.cu_offset)
                if has_views:
                    view = loc_list[loc_entry_count]
                    postfix = ' (start == end)' if entry.begin_offset == entry.end_offset and view.begin == view.end else ''
                    self._emitline('    %08x v%015x v%015x views at %08x for:' % (entry.entry_offset, view.begin, view.end, view.entry_offset))
                    self._emitline('             %016x %016x %s%s' % (begin_offset, end_offset, expr, postfix))
                    loc_entry_count += 1
                else:
                    postfix = ' (start == end)' if entry.begin_offset == entry.end_offset else ''
                    self._emitline(line_template % (entry.entry_offset, begin_offset, end_offset, expr, postfix))
            elif isinstance(entry, LocBaseAddressEntry):
                base_ip = entry.base_address
                self._emitline('    %08x %016x (base address)' % (entry.entry_offset, entry.base_address))
    last = loc_list[-1]
    self._emitline('    %08x <End of list>' % (last.entry_offset + last.entry_length))