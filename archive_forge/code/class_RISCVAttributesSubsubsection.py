from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class RISCVAttributesSubsubsection(AttributesSubsubsection):
    """ Subsubsection of an ELF .riscv.attributes subsection.
    """

    def __init__(self, stream, structs, offset):
        super(RISCVAttributesSubsubsection, self).__init__(stream, structs, offset, RISCVAttribute)