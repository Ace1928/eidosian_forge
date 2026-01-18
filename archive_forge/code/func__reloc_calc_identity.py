from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def _reloc_calc_identity(value, sym_value, offset, addend=0):
    return value