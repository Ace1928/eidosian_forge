import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
class LocationListsPair(object):
    """For those binaries that contain both a debug_loc and a debug_loclists section,
    it holds a LocationLists object for both and forwards API calls to the right one.
    """

    def __init__(self, streamv4, streamv5, structs, dwarfinfo=None):
        self._loc = LocationLists(streamv4, structs, 4, dwarfinfo)
        self._loclists = LocationLists(streamv5, structs, 5, dwarfinfo)

    def get_location_list_at_offset(self, offset, die=None):
        """See LocationLists.get_location_list_at_offset().
        """
        if die is None:
            raise DWARFError('For this binary, "die" needs to be provided')
        section = self._loclists if die.cu.header.version >= 5 else self._loc
        return section.get_location_list_at_offset(offset, die)

    def iter_location_lists(self):
        """Tricky proposition, since the structure of loc and loclists
        is not identical. A realistic readelf implementation needs to be aware of both
        """
        raise DWARFError('Iterating through two sections is not supported')

    def iter_CUs(self):
        """See LocationLists.iter_CUs()

        There are no CUs in DWARFv4 sections.
        """
        raise DWARFError('Iterating through two sections is not supported')