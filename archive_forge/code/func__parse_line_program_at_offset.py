import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def _parse_line_program_at_offset(self, offset, structs):
    """ Given an offset to the .debug_line section, parse the line program
            starting at this offset in the section and return it.
            structs is the DWARFStructs object used to do this parsing.
        """
    if offset in self._linetable_cache:
        return self._linetable_cache[offset]
    lineprog_header = struct_parse(structs.Dwarf_lineprog_header, self.debug_line_sec.stream, offset)

    def resolve_strings(self, lineprog_header, format_field, data_field):
        if lineprog_header.get(format_field, False):
            data = lineprog_header[data_field]
            for field in lineprog_header[format_field]:

                def replace_value(data, content_type, replacer):
                    for entry in data:
                        entry[content_type] = replacer(entry[content_type])
                if field.form == 'DW_FORM_line_strp':
                    replace_value(data, field.content_type, self.get_string_from_linetable)
                elif field.form == 'DW_FORM_strp':
                    replace_value(data, field.content_type, self.get_string_from_table)
                elif field.form in ('DW_FORM_strp_sup', 'DW_FORM_GNU_strp_alt'):
                    if self.supplementary_dwarfinfo:
                        replace_value(data, field.content_type, self.supplementary_dwarfinfo.get_string_fromtable)
                    else:
                        replace_value(data, field.content_type, lambda x: str(x))
                elif field.form in ('DW_FORM_strp_sup', 'DW_FORM_strx', 'DW_FORM_strx1', 'DW_FORM_strx2', 'DW_FORM_strx3', 'DW_FORM_strx4'):
                    raise NotImplementedError()
    resolve_strings(self, lineprog_header, 'directory_entry_format', 'directories')
    resolve_strings(self, lineprog_header, 'file_name_entry_format', 'file_names')
    if lineprog_header.get('directories', False):
        lineprog_header.include_directory = tuple((d.DW_LNCT_path for d in lineprog_header.directories))
    if lineprog_header.get('file_names', False):
        lineprog_header.file_entry = tuple((Container(**{'name': e.get('DW_LNCT_path'), 'dir_index': e.get('DW_LNCT_directory_index'), 'mtime': e.get('DW_LNCT_timestamp'), 'length': e.get('DW_LNCT_size')}) for e in lineprog_header.file_names))
    end_offset = offset + lineprog_header['unit_length'] + structs.initial_length_field_size()
    lineprogram = LineProgram(header=lineprog_header, stream=self.debug_line_sec.stream, structs=structs, program_start_offset=self.debug_line_sec.stream.tell(), program_end_offset=end_offset)
    self._linetable_cache[offset] = lineprogram
    return lineprogram