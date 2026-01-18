from ..construct import CString
from ..common.utils import struct_parse
from .constants import SH_FLAGS
from .notes import iter_notes
class NoteSegment(Segment):
    """ NOTE segment. Knows how to parse notes.
    """

    def __init__(self, header, stream, elffile):
        super(NoteSegment, self).__init__(header, stream)
        self.elffile = elffile

    def iter_notes(self):
        """ Yield all the notes in the segment.  Each result is a dictionary-
            like object with "n_name", "n_type", and "n_desc" fields, amongst
            others.
        """
        return iter_notes(self.elffile, self['p_offset'], self['p_filesz'])