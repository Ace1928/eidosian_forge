import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def _parse_locview_pairs(self, locviews):
    stream = self.stream
    list_offset = locviews.get(stream.tell(), None)
    pairs = []
    if list_offset is not None:
        while stream.tell() < list_offset:
            pair = struct_parse(self.structs.Dwarf_locview_pair, stream)
            pairs.append(LocationViewPair(pair.entry_offset, pair.begin, pair.end))
        assert stream.tell() == list_offset
    return pairs