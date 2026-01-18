import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
@staticmethod
def _attribute_has_loc_list(attr, dwarf_version):
    return (dwarf_version < 4 and attr.form in ('DW_FORM_data1', 'DW_FORM_data2', 'DW_FORM_data4', 'DW_FORM_data8') and (not attr.name == 'DW_AT_const_value') or attr.form in ('DW_FORM_sec_offset', 'DW_FORM_loclistx')) and (not LocationParser._attribute_is_constant(attr, dwarf_version))