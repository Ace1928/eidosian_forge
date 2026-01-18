from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _InitialLength(name):
    return _InitialLengthAdapter(Struct(name, self.Dwarf_uint32('first'), If(lambda ctx: ctx.first == 4294967295, self.Dwarf_uint64('second'), elsevalue=None)))