import os
from ..construct.macros import UBInt32, UBInt64, ULInt32, ULInt64, Array
from ..common.exceptions import DWARFError
from ..common.utils import preserve_stream_pos, struct_parse
def _resolve_via_offset_table(stream, cu, index, base_attribute_name):
    """Given an index in the offset table and directions where to find it,
    retrieves an offset. Works for loclists, rnglists.

    The DWARF offset bitness of the CU block in the section matches that
    of the CU record in dwarf_info. See DWARFv5 standard, section 7.4.

    This is used for translating DW_FORM_loclistx, DW_FORM_rnglistx
    via the offset table in the respective section.
    """
    base_offset = _get_base_offset(cu, base_attribute_name)
    offset_size = 4 if cu.structs.dwarf_format == 32 else 8
    with preserve_stream_pos(stream):
        return base_offset + struct_parse(cu.structs.Dwarf_offset(''), stream, base_offset + index * offset_size)