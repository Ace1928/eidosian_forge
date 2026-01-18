from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def _arm_reloc_calc_sym_plus_value_pcrel(value, sym_value, offset, addend=0):
    return sym_value // 4 + value - offset // 4