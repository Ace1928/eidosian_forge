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
def _symbol_version(self, nsym):
    """ Return a dict containing information on the
            or None if no version information is available
        """
    self._init_versioninfo()
    symbol_version = dict.fromkeys(('index', 'name', 'filename', 'hidden'))
    if not self._versioninfo['versym'] or nsym >= self._versioninfo['versym'].num_symbols():
        return None
    symbol = self._versioninfo['versym'].get_symbol(nsym)
    index = symbol.entry['ndx']
    if not index in ('VER_NDX_LOCAL', 'VER_NDX_GLOBAL'):
        index = int(index)
        if self._versioninfo['type'] == 'GNU':
            if index & 32768:
                index &= ~32768
                symbol_version['hidden'] = True
        if self._versioninfo['verdef'] and index <= self._versioninfo['verdef'].num_versions():
            _, verdaux_iter = self._versioninfo['verdef'].get_version(index)
            symbol_version['name'] = next(verdaux_iter).name
        else:
            verneed, vernaux = self._versioninfo['verneed'].get_version(index)
            symbol_version['name'] = vernaux.name
            symbol_version['filename'] = verneed.name
    symbol_version['index'] = index
    return symbol_version