from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _assert_FDE_instruction(instr):
    dwarf_assert(isinstance(entry, FDE), 'Unexpected instruction "%s" for a CIE' % instr)