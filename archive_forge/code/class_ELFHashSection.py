import struct
from ..common.utils import struct_parse
from .sections import Section
class ELFHashSection(Section, ELFHashTable):
    """ Section representation of an ELF hash table. In regular ELF files, this
        allows us to use the common functions defined on Section objects when
        dealing with the hash table.
    """

    def __init__(self, header, name, elffile, symboltable):
        Section.__init__(self, header, name, elffile)
        ELFHashTable.__init__(self, elffile, self['sh_offset'], symboltable)