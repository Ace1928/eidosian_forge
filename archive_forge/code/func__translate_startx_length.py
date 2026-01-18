import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def _translate_startx_length(e, cu):
    start_offset = cu.dwarfinfo.get_addr(cu, e.start_index)
    return LocationEntry(e.entry_offset, e.entry_length, start_offset, start_offset + e.length, e.loc_expr, True)