from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_nameLUT_header(self):
    self.Dwarf_nameLUT_header = Struct('Dwarf_nameLUT_header', self.Dwarf_initial_length('unit_length'), self.Dwarf_uint16('version'), self.Dwarf_offset('debug_info_offset'), self.Dwarf_length('debug_info_length'))