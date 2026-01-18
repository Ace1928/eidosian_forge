import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
@staticmethod
def _attribute_is_loclistptr_class(attr):
    return attr.name in ('DW_AT_location', 'DW_AT_string_length', 'DW_AT_const_value', 'DW_AT_return_addr', 'DW_AT_data_member_location', 'DW_AT_frame_base', 'DW_AT_segment', 'DW_AT_static_link', 'DW_AT_use_location', 'DW_AT_vtable_elem_location', 'DW_AT_call_value', 'DW_AT_GNU_call_site_value', 'DW_AT_GNU_call_site_target', 'DW_AT_GNU_call_site_data_value', 'DW_AT_call_target', 'DW_AT_call_target_clobbered', 'DW_AT_call_data_location', 'DW_AT_call_data_value', 'DW_AT_upper_bound', 'DW_AT_count')