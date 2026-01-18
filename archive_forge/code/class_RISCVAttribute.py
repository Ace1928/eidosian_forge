from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class RISCVAttribute(Attribute):
    """ Attribute of an ELF .riscv.attributes section.
    """

    def __init__(self, structs, stream):
        super(RISCVAttribute, self).__init__(struct_parse(structs.Elf_RiscV_Attribute_Tag, stream))
        if self.tag in ('TAG_FILE', 'TAG_SECTION', 'TAG_SYMBOL'):
            self.value = struct_parse(structs.Elf_word('value'), stream)
            if self.tag != 'TAG_FILE':
                self.extra = []
                s_number = struct_parse(structs.Elf_uleb128('s_number'), stream)
                while s_number != 0:
                    self.extra.append(s_number)
                    s_number = struct_parse(structs.Elf_uleb128('s_number'), stream)
        elif self.tag == 'TAG_ARCH':
            self.value = struct_parse(structs.Elf_ntbs('value', encoding='utf-8'), stream)
        else:
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)