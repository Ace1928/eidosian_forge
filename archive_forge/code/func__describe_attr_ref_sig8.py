from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _describe_attr_ref_sig8(attr, die, section_offset):
    return 'signature: %s' % _format_hex(attr.value)