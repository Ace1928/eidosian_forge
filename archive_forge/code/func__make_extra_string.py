from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _make_extra_string(s=''):
    """ Create an extra function that just returns a constant string.
    """

    def extra(attr, die, section_offset):
        return s
    return extra