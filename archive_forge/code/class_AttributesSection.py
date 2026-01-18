from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class AttributesSection(Section):
    """ ELF attributes section.
    """

    def __init__(self, header, name, elffile, subsection):
        super(AttributesSection, self).__init__(header, name, elffile)
        self.subsection = subsection
        fv = struct_parse(self.structs.Elf_byte('format_version'), self.stream, self['sh_offset'])
        elf_assert(chr(fv) == 'A', "Unknown attributes version %s, expecting 'A'." % chr(fv))
        self.subsec_start = self.stream.tell()

    def iter_subsections(self, vendor_name=None):
        """ Yield all subsections (limit to |vendor_name| if specified).
        """
        for subsec in self._make_subsections():
            if vendor_name is None or subsec['vendor_name'] == vendor_name:
                yield subsec

    @property
    def num_subsections(self):
        """ Number of subsections in the section.
        """
        return sum((1 for _ in self.iter_subsections()))

    @property
    def subsections(self):
        """ List of all subsections in the section.
        """
        return list(self.iter_subsections())

    def _make_subsections(self):
        """ Create all subsections for this section.
        """
        end = self['sh_offset'] + self.data_size
        self.stream.seek(self.subsec_start)
        while self.stream.tell() != end:
            subsec = self.subsection(self.stream, self.structs, self.stream.tell())
            self.stream.seek(self.subsec_start + subsec['length'])
            yield subsec