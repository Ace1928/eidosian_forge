from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def _parse_DIE(self):
    """ Parses the DIE info from the section, based on the abbreviation
            table of the CU
        """
    structs = self.cu.structs
    self.abbrev_code = struct_parse(structs.Dwarf_uleb128(''), self.stream, self.offset)
    if self.abbrev_code == 0:
        self.size = self.stream.tell() - self.offset
        return
    abbrev_decl = self.cu.get_abbrev_table().get_abbrev(self.abbrev_code)
    self.tag = abbrev_decl['tag']
    self.has_children = abbrev_decl.has_children()
    for spec in abbrev_decl['attr_spec']:
        form = spec.form
        name = spec.name
        attr_offset = self.stream.tell()
        indirection_length = 0
        if form == 'DW_FORM_implicit_const':
            value = spec.value
            raw_value = value
        elif form == 'DW_FORM_indirect':
            form, raw_value, indirection_length = self._resolve_indirect()
            value = self._translate_attr_value(form, raw_value)
        else:
            raw_value = struct_parse(structs.Dwarf_dw_form[form], self.stream)
            value = self._translate_attr_value(form, raw_value)
        self.attributes[name] = AttributeValue(name=name, form=form, value=value, raw_value=raw_value, offset=attr_offset, indirection_length=indirection_length)
    self.size = self.stream.tell() - self.offset