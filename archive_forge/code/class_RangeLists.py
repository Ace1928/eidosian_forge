import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
class RangeLists(object):
    """ A single range list is a Python list consisting of RangeEntry or
        BaseAddressEntry objects.

        Since v0.29, two new parameters - version and dwarfinfo

        version is used to distinguish DWARFv5 rnglists section from
        the DWARF<=4 ranges section. Only the 4/5 distinction matters.

        The dwarfinfo is needed for enumeration, because enumeration
        requires scanning the DIEs, because ranges may overlap, even on DWARF<=4
    """

    def __init__(self, stream, structs, version, dwarfinfo):
        self.stream = stream
        self.structs = structs
        self._max_addr = 2 ** (self.structs.address_size * 8) - 1
        self.version = version
        self._dwarfinfo = dwarfinfo

    def get_range_list_at_offset(self, offset, cu=None):
        """ Get a range list at the given offset in the section.

            The cu argument is necessary if the ranges section is a
            DWARFv5 debug_rnglists one, and the target rangelist
            contains indirect encodings
        """
        self.stream.seek(offset, os.SEEK_SET)
        return self._parse_range_list_from_stream(cu)

    def get_range_list_at_offset_ex(self, offset):
        """Get a DWARF v5 range list, addresses and offsets unresolved,
        at the given offset in the section
        """
        return struct_parse(self.structs.Dwarf_rnglists_entries, self.stream, offset)

    def iter_range_lists(self):
        """ Yields all range lists found in the section according to readelf rules.
        Scans the DIEs for rangelist offsets, then pulls those.
        Returned rangelists are always translated into lists of BaseAddressEntry/RangeEntry objects.
        """
        ver5 = self.version >= 5
        cu_map = {die.attributes['DW_AT_ranges'].value: cu for cu in self._dwarfinfo.iter_CUs() for die in cu.iter_DIEs() if 'DW_AT_ranges' in die.attributes and (cu['version'] >= 5) == ver5}
        all_offsets = list(cu_map.keys())
        all_offsets.sort()
        for offset in all_offsets:
            yield self.get_range_list_at_offset(offset, cu_map[offset])

    def iter_CUs(self):
        """For DWARF5 returns an array of objects, where each one has an array of offsets
        """
        if self.version < 5:
            raise DWARFError('CU iteration in rnglists is not supported with DWARF<5')
        structs = next(self._dwarfinfo.iter_CUs()).structs
        return _iter_CUs_in_section(self.stream, structs, structs.Dwarf_rnglists_CU_header)

    def iter_CU_range_lists_ex(self, cu):
        """For DWARF5, returns untranslated rangelists in the CU, where CU comes from iter_CUs above
        """
        stream = self.stream
        stream.seek(cu.offset_table_offset + (64 if cu.is64 else 32) * cu.offset_count)
        while stream.tell() < cu.offset_after_length + cu.unit_length:
            yield struct_parse(self.structs.Dwarf_rnglists_entries, stream)

    def translate_v5_entry(self, entry, cu):
        """Translates entries in a DWARFv5 rangelist from raw parsed format to 
        a list of BaseAddressEntry/RangeEntry, using the CU
        """
        return entry_translate[entry.entry_type](entry, cu)

    def _parse_range_list_from_stream(self, cu):
        if self.version >= 5:
            return list((entry_translate[entry.entry_type](entry, cu) for entry in struct_parse(self.structs.Dwarf_rnglists_entries, self.stream)))
        else:
            lst = []
            while True:
                entry_offset = self.stream.tell()
                begin_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
                end_offset = struct_parse(self.structs.Dwarf_target_addr(''), self.stream)
                if begin_offset == 0 and end_offset == 0:
                    break
                elif begin_offset == self._max_addr:
                    lst.append(BaseAddressEntry(entry_offset=entry_offset, base_address=end_offset))
                else:
                    lst.append(RangeEntry(entry_offset=entry_offset, entry_length=self.stream.tell() - entry_offset, begin_offset=begin_offset, end_offset=end_offset, is_absolute=False))
            return lst