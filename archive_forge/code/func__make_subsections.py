from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def _make_subsections(self):
    """ Create all subsections for this section.
        """
    end = self['sh_offset'] + self.data_size
    self.stream.seek(self.subsec_start)
    while self.stream.tell() != end:
        subsec = self.subsection(self.stream, self.structs, self.stream.tell())
        self.stream.seek(self.subsec_start + subsec['length'])
        yield subsec