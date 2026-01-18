from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_structs(self):
    if self.little_endian:
        self.Dwarf_uint8 = ULInt8
        self.Dwarf_uint16 = ULInt16
        self.Dwarf_uint32 = ULInt32
        self.Dwarf_uint64 = ULInt64
        self.Dwarf_offset = ULInt32 if self.dwarf_format == 32 else ULInt64
        self.Dwarf_length = ULInt32 if self.dwarf_format == 32 else ULInt64
        self.Dwarf_target_addr = ULInt32 if self.address_size == 4 else ULInt64
        self.Dwarf_int8 = SLInt8
        self.Dwarf_int16 = SLInt16
        self.Dwarf_int32 = SLInt32
        self.Dwarf_int64 = SLInt64
    else:
        self.Dwarf_uint8 = UBInt8
        self.Dwarf_uint16 = UBInt16
        self.Dwarf_uint32 = UBInt32
        self.Dwarf_uint64 = UBInt64
        self.Dwarf_offset = UBInt32 if self.dwarf_format == 32 else UBInt64
        self.Dwarf_length = UBInt32 if self.dwarf_format == 32 else UBInt64
        self.Dwarf_target_addr = UBInt32 if self.address_size == 4 else UBInt64
        self.Dwarf_int8 = SBInt8
        self.Dwarf_int16 = SBInt16
        self.Dwarf_int32 = SBInt32
        self.Dwarf_int64 = SBInt64
    self._create_initial_length()
    self._create_leb128()
    self._create_cu_header()
    self._create_abbrev_declaration()
    self._create_dw_form()
    self._create_lineprog_header()
    self._create_callframe_entry_headers()
    self._create_aranges_header()
    self._create_nameLUT_header()
    self._create_string_offsets_table_header()
    self._create_address_table_header()
    self._create_loclists_parsers()
    self._create_rnglists_parsers()
    self._create_debugsup()
    self._create_gnu_debugaltlink()