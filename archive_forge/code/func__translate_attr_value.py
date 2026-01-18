from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def _translate_attr_value(self, form, raw_value):
    """ Translate a raw attr value according to the form
        """
    translate_indirect = self.cu.has_top_DIE() or self.offset != self.cu.cu_die_offset
    value = None
    if form == 'DW_FORM_strp':
        with preserve_stream_pos(self.stream):
            value = self.dwarfinfo.get_string_from_table(raw_value)
    elif form == 'DW_FORM_line_strp':
        with preserve_stream_pos(self.stream):
            value = self.dwarfinfo.get_string_from_linetable(raw_value)
    elif form in ('DW_FORM_GNU_strp_alt', 'DW_FORM_strp_sup'):
        if self.dwarfinfo.supplementary_dwarfinfo:
            return self.dwarfinfo.supplementary_dwarfinfo.get_string_from_table(raw_value)
        else:
            value = raw_value
    elif form == 'DW_FORM_flag':
        value = not raw_value == 0
    elif form == 'DW_FORM_flag_present':
        value = True
    elif form in ('DW_FORM_addrx', 'DW_FORM_addrx1', 'DW_FORM_addrx2', 'DW_FORM_addrx3', 'DW_FORM_addrx4') and translate_indirect:
        value = self.cu.dwarfinfo.get_addr(self.cu, raw_value)
    elif form in ('DW_FORM_strx', 'DW_FORM_strx1', 'DW_FORM_strx2', 'DW_FORM_strx3', 'DW_FORM_strx4') and translate_indirect:
        stream = self.dwarfinfo.debug_str_offsets_sec.stream
        base_offset = _get_base_offset(self.cu, 'DW_AT_str_offsets_base')
        offset_size = 4 if self.cu.structs.dwarf_format == 32 else 8
        with preserve_stream_pos(stream):
            str_offset = struct_parse(self.cu.structs.Dwarf_offset(''), stream, base_offset + raw_value * offset_size)
        value = self.dwarfinfo.get_string_from_table(str_offset)
    elif form == 'DW_FORM_loclistx' and translate_indirect:
        value = _resolve_via_offset_table(self.dwarfinfo.debug_loclists_sec.stream, self.cu, raw_value, 'DW_AT_loclists_base')
    elif form == 'DW_FORM_rnglistx' and translate_indirect:
        value = _resolve_via_offset_table(self.dwarfinfo.debug_rnglists_sec.stream, self.cu, raw_value, 'DW_AT_rnglists_base')
    else:
        value = raw_value
    return value