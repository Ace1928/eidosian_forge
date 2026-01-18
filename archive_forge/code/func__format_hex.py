from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _format_hex(n):
    return '0x%x' % n if n != 0 else '0'