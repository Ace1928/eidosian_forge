import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
class RangeListsPair(object):
    """For those binaries that contain both a debug_ranges and a debug_rnglists section,
    it holds a RangeLists object for both and forwards API calls to the right one based
    on the CU version.
    """

    def __init__(self, streamv4, streamv5, structs, dwarfinfo=None):
        self._ranges = RangeLists(streamv4, structs, 4, dwarfinfo)
        self._rnglists = RangeLists(streamv5, structs, 5, dwarfinfo)

    def get_range_list_at_offset(self, offset, cu=None):
        """Forwards the call to either v4 section or v5 one,
        depending on DWARF version in the CU.
        """
        if cu is None:
            raise DWARFError('For this binary, "cu" needs to be provided')
        section = self._rnglists if cu.header.version >= 5 else self._ranges
        return section.get_range_list_at_offset(offset, cu)

    def get_range_list_at_offset_ex(self, offset):
        """Gets an untranslated v5 rangelist from the v5 section.
        """
        return self._rnglists.get_range_list_at_offset_ex(offset)

    def iter_range_lists(self):
        """Tricky proposition, since the structure of ranges and rnglists
        is not identical. A realistic readelf implementation needs to be aware of both.
        """
        raise DWARFError('Iterating through two sections is not supported')

    def iter_CUs(self):
        """See RangeLists.iter_CUs()
        
        CU structure is only present in DWARFv5 rnglists sections. A well written
        section dumper should check if one is present.
        """
        return self._rnglists.iter_CUs()

    def iter_CU_range_lists_ex(self, cu):
        """See RangeLists.iter_CU_range_lists_ex()

        CU structure is only present in DWARFv5 rnglists sections. A well written
        section dumper should check if one is present.
        """
        return self._rnglists.iter_CU_range_lists_ex(cu)

    def translate_v5_entry(self, entry, cu):
        """Forwards a V5 entry translation request to the V5 section
        """
        return self._rnglists.translate_v5_entry(entry, cu)