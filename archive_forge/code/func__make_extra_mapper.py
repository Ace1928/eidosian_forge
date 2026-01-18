from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def _make_extra_mapper(mapping, default, default_interpolate_value=False):
    """ Create a mapping function from attribute parameters to an extra
        value that should be displayed.
    """

    def mapper(attr, die, section_offset):
        if default_interpolate_value:
            d = default % attr.value
        else:
            d = default
        return mapping.get(attr.value, d)
    return mapper