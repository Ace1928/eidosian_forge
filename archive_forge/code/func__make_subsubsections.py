from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def _make_subsubsections(self):
    """ Create all subsubsections for this subsection.
        """
    end = self.offset + self['length']
    self.stream.seek(self.subsubsec_start)
    while self.stream.tell() != end:
        subsubsec = self.subsubsection(self.stream, self.structs, self.stream.tell())
        self.stream.seek(self.subsubsec_start + subsubsec.header.value)
        yield subsubsec