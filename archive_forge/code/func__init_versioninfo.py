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
def _init_versioninfo(self):
    """ Search and initialize informations about version related sections
            and the kind of versioning used (GNU or Solaris).
        """
    if self._versioninfo is not None:
        return
    self._versioninfo = {'versym': None, 'verdef': None, 'verneed': None, 'type': None}
    for section in self.elffile.iter_sections():
        if isinstance(section, GNUVerSymSection):
            self._versioninfo['versym'] = section
        elif isinstance(section, GNUVerDefSection):
            self._versioninfo['verdef'] = section
        elif isinstance(section, GNUVerNeedSection):
            self._versioninfo['verneed'] = section
        elif isinstance(section, DynamicSection):
            for tag in section.iter_tags():
                if tag['d_tag'] == 'DT_VERSYM':
                    self._versioninfo['type'] = 'GNU'
                    break
    if not self._versioninfo['type'] and (self._versioninfo['verneed'] or self._versioninfo['verdef']):
        self._versioninfo['type'] = 'Solaris'