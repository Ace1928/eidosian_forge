from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class RelrRelocationSection(Section, RelrRelocationTable):
    """ ELF RELR relocation section. Serves as a collection of RELR relocation entries.
    """

    def __init__(self, header, name, elffile):
        Section.__init__(self, header, name, elffile)
        RelrRelocationTable.__init__(self, self.elffile, self['sh_offset'], self['sh_size'], self['sh_entsize'])