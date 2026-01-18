import os
import collections
from collections import OrderedDict
from collections.abc import Mapping
from ..common.utils import struct_parse
from bisect import bisect_right
import math
from ..construct import CString, Struct, If
class NameLUT(Mapping):
    """
    A "Name LUT" holds any of the tables specified by .debug_pubtypes or
    .debug_pubnames sections. This is basically a dictionary where the key is
    the symbol name (either a public variable, function or a type), and the
    value is the tuple (cu_offset, die_offset) corresponding to the variable.
    The die_offset is an absolute offset (meaning, it can be used to search the
    CU by iterating until a match is obtained).

    An ordered dictionary is used to preserve the CU order (i.e, items are
    stored on a per-CU basis (as it was originally in the .debug_* section).

    Usage:

    The NameLUT walks and talks like a dictionary and hence it can be used as
    such. Some examples below:

    # get the pubnames (a NameLUT from DWARF info).
    pubnames = dwarf_info.get_pubnames()

    # lookup a variable.
    entry1 = pubnames["var_name1"]
    entry2 = pubnames.get("var_name2", default=<default_var>)
    print(entry2.cu_ofs)
    ...

    # iterate over items.
    for (name, entry) in pubnames.items():
      # do stuff with name, entry.cu_ofs, entry.die_ofs

    # iterate over items on a per-CU basis.
    import itertools
    for cu_ofs, item_list in itertools.groupby(pubnames.items(),
        key = lambda x: x[1].cu_ofs):
      # items are now grouped by cu_ofs.
      # item_list is an iterator yeilding NameLUTEntry'ies belonging
      # to cu_ofs.
      # We can parse the CU at cu_offset and use the parsed CU results
      # to parse the pubname DIEs in the CU listed by item_list.
      for item in item_list:
        # work with item which is part of the CU with cu_ofs.

    """

    def __init__(self, stream, size, structs):
        self._stream = stream
        self._size = size
        self._structs = structs
        self._entries = None
        self._cu_headers = None

    def get_entries(self):
        """
        Returns the parsed NameLUT entries. The returned object is a dictionary
        with the symbol name as the key and NameLUTEntry(cu_ofs, die_ofs) as
        the value.

        This is useful when dealing with very large ELF files with millions of
        entries. The returned entries can be pickled to a file and restored by
        calling set_entries on subsequent loads.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return self._entries

    def set_entries(self, entries, cu_headers):
        """
        Set the NameLUT entries from an external source. The input is a
        dictionary with the symbol name as the key and NameLUTEntry(cu_ofs,
        die_ofs) as the value.

        This option is useful when dealing with very large ELF files with
        millions of entries. The entries can be parsed once and pickled to a
        file and can be restored via this function on subsequent loads.
        """
        self._entries = entries
        self._cu_headers = cu_headers

    def __len__(self):
        """
        Returns the number of entries in the NameLUT.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return len(self._entries)

    def __getitem__(self, name):
        """
        Returns a namedtuple - NameLUTEntry(cu_ofs, die_ofs) - that corresponds
        to the given symbol name.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return self._entries.get(name)

    def __iter__(self):
        """
        Returns an iterator to the NameLUT dictionary.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return iter(self._entries)

    def items(self):
        """
        Returns the NameLUT dictionary items.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return self._entries.items()

    def get(self, name, default=None):
        """
        Returns NameLUTEntry(cu_ofs, die_ofs) for the provided symbol name or
        None if the symbol does not exist in the corresponding section.
        """
        if self._entries is None:
            self._entries, self._cu_headers = self._get_entries()
        return self._entries.get(name, default)

    def get_cu_headers(self):
        """
        Returns all CU headers. Mainly required for readelf.
        """
        if self._cu_headers is None:
            self._entries, self._cu_headers = self._get_entries()
        return self._cu_headers

    def _get_entries(self):
        """
        Parse the (name, cu_ofs, die_ofs) information from this section and
        store as a dictionary.
        """
        self._stream.seek(0)
        entries = OrderedDict()
        cu_headers = []
        offset = 0
        entry_struct = Struct('Dwarf_offset_name_pair', self._structs.Dwarf_offset('die_ofs'), If(lambda ctx: ctx['die_ofs'], CString('name')))
        while offset < self._size:
            namelut_hdr = struct_parse(self._structs.Dwarf_nameLUT_header, self._stream, offset)
            cu_headers.append(namelut_hdr)
            offset = offset + namelut_hdr.unit_length + self._structs.initial_length_field_size()
            hdr_cu_ofs = namelut_hdr.debug_info_offset
            while True:
                entry = struct_parse(entry_struct, self._stream)
                if entry.die_ofs == 0:
                    break
                entries[entry.name.decode('utf-8')] = NameLUTEntry(cu_ofs=hdr_cu_ofs, die_ofs=hdr_cu_ofs + entry.die_ofs)
        return (entries, cu_headers)