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
def _dump_debug_locsection(self, di, loc_lists_sec):
    """ Dump the location lists from .debug_loc/.debug_loclists section
        """
    ver5 = loc_lists_sec.version >= 5
    section_name = (di.debug_loclists_sec if ver5 else di.debug_loc_sec).name
    cu_map = dict()
    for cu in di.iter_CUs():
        for die in cu.iter_DIEs():
            for key in die.attributes:
                attr = die.attributes[key]
                if LocationParser.attribute_has_location(attr, cu['version']) and LocationParser._attribute_has_loc_list(attr, cu['version']):
                    cu_map[attr.value] = cu
    addr_size = di.config.default_address_size
    addr_width = addr_size * 2
    line_template = '    %%08x %%0%dx %%0%dx %%s%%s' % (addr_width, addr_width)
    loc_lists = list(loc_lists_sec.iter_location_lists())
    if len(loc_lists) == 0:
        self._emitline("\nSection '%s' has no debugging data." % (section_name,))
        return
    self._emitline('Contents of the %s section:\n' % (section_name,))
    self._emitline('    Offset   Begin            End              Expression')
    for loc_list in loc_lists:
        self._dump_loclist(loc_list, line_template, cu_map)