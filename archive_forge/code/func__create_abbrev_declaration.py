from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_abbrev_declaration(self):
    self.Dwarf_abbrev_declaration = Struct('Dwarf_abbrev_entry', Enum(self.Dwarf_uleb128('tag'), **ENUM_DW_TAG), Enum(self.Dwarf_uint8('children_flag'), **ENUM_DW_CHILDREN), RepeatUntilExcluding(lambda obj, ctx: obj.name == 'DW_AT_null' and obj.form == 'DW_FORM_null', Struct('attr_spec', Enum(self.Dwarf_uleb128('name'), **ENUM_DW_AT), Enum(self.Dwarf_uleb128('form'), **ENUM_DW_FORM), If(lambda ctx: ctx['form'] == 'DW_FORM_implicit_const', self.Dwarf_sleb128('value')))))