import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
@staticmethod
def _attribute_is_constant(attr, dwarf_version):
    return (dwarf_version >= 3 and attr.name == 'DW_AT_data_member_location' or attr.name in ('DW_AT_upper_bound', 'DW_AT_count')) and attr.form in ('DW_FORM_data1', 'DW_FORM_data2', 'DW_FORM_data4', 'DW_FORM_data8', 'DW_FORM_sdata', 'DW_FORM_udata')