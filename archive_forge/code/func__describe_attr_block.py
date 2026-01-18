from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _describe_attr_block(attr, die, section_offset):
    s = '%s byte block: ' % len(attr.value)
    s += ' '.join(('%x' % item for item in attr.value)) + ' '
    return s