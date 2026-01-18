from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class AttributesSubsubsection(Section):
    """ Subsubsection of an ELF attribute section's subsection.
    """

    def __init__(self, stream, structs, offset, attribute):
        self.stream = stream
        self.offset = offset
        self.structs = structs
        self.attribute = attribute
        self.header = self.attribute(self.structs, self.stream)
        self.attr_start = self.stream.tell()

    def iter_attributes(self, tag=None):
        """ Yield all attributes (limit to |tag| if specified).
        """
        for attribute in self._make_attributes():
            if tag is None or attribute.tag == tag:
                yield attribute

    @property
    def num_attributes(self):
        """ Number of attributes in the subsubsection.
        """
        return sum((1 for _ in self.iter_attributes())) + 1

    @property
    def attributes(self):
        """ List of all attributes in the subsubsection.
        """
        return [self.header] + list(self.iter_attributes())

    def _make_attributes(self):
        """ Create all attributes for this subsubsection except the first one
            which is the header.
        """
        end = self.offset + self.header.value
        self.stream.seek(self.attr_start)
        while self.stream.tell() != end:
            yield self.attribute(self.structs, self.stream)

    def __repr__(self):
        s = '<%s (%s): %d bytes>'
        return s % (self.__class__.__name__, self.header.tag[4:], self.header.value)