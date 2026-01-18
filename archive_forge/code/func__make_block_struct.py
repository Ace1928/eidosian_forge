from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _make_block_struct(self, length_field):
    """ Create a struct for DW_FORM_block<size>
        """
    return PrefixedArray(subcon=self.Dwarf_uint8('elem'), length_field=length_field(''))