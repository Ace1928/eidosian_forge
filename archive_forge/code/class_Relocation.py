from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class Relocation(object):
    """ Relocation object - representing a single relocation entry. Allows
        dictionary-like access to the entry's fields.

        Can be either a REL or RELA relocation.
    """

    def __init__(self, entry, elffile):
        self.entry = entry
        self.elffile = elffile

    def is_RELA(self):
        """ Is this a RELA relocation? If not, it's REL.
        """
        return 'r_addend' in self.entry

    def __getitem__(self, name):
        """ Dict-like access to entries
        """
        return self.entry[name]

    def __repr__(self):
        return '<Relocation (%s): %s>' % ('RELA' if self.is_RELA() else 'REL', self.entry)

    def __str__(self):
        return self.__repr__()