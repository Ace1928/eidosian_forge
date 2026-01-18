from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class ARMAttribute(Attribute):
    """ ARM attribute object - representing a build attribute of ARM ELF files.
    """

    def __init__(self, structs, stream):
        super(ARMAttribute, self).__init__(struct_parse(structs.Elf_Arm_Attribute_Tag, stream))
        if self.tag in ('TAG_FILE', 'TAG_SECTION', 'TAG_SYMBOL'):
            self.value = struct_parse(structs.Elf_word('value'), stream)
            if self.tag != 'TAG_FILE':
                self.extra = []
                s_number = struct_parse(structs.Elf_uleb128('s_number'), stream)
                while s_number != 0:
                    self.extra.append(s_number)
                    s_number = struct_parse(structs.Elf_uleb128('s_number'), stream)
        elif self.tag in ('TAG_CPU_RAW_NAME', 'TAG_CPU_NAME', 'TAG_CONFORMANCE'):
            self.value = struct_parse(structs.Elf_ntbs('value', encoding='utf-8'), stream)
        elif self.tag == 'TAG_COMPATIBILITY':
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)
            self.extra = struct_parse(structs.Elf_ntbs('vendor_name', encoding='utf-8'), stream)
        elif self.tag == 'TAG_ALSO_COMPATIBLE_WITH':
            self.value = ARMAttribute(structs, stream)
            if type(self.value.value) is not str:
                nul = struct_parse(structs.Elf_byte('nul'), stream)
                elf_assert(nul == 0, 'Invalid terminating byte %r, expecting NUL.' % nul)
        else:
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)