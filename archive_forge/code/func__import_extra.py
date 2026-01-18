from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _import_extra(attr, die, section_offset):
    if attr.form == 'DW_FORM_ref_addr':
        ref_die_offset = section_offset + attr.value
    else:
        ref_die_offset = attr.value + die.cu.cu_offset
    for cu in die.dwarfinfo.iter_CUs():
        if cu['unit_length'] + cu.cu_offset > ref_die_offset >= cu.cu_offset:
            with preserve_stream_pos(die.stream):
                ref_die = DIE(cu, die.stream, ref_die_offset)
            return '[Abbrev Number: %s (%s)]' % (ref_die.abbrev_code, ref_die.tag)
    return '[unknown]'