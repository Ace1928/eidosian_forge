import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class SequentialSeqFileRandomAccess(SeqFileRandomAccess):
    """Random access to a simple sequential sequence file."""

    def __init__(self, filename, format):
        """Initialize the class."""
        SeqFileRandomAccess.__init__(self, filename, format)
        marker = {'ace': b'CO ', 'embl': b'ID ', 'fasta': b'>', 'genbank': b'LOCUS ', 'gb': b'LOCUS ', 'imgt': b'ID ', 'phd': b'BEGIN_SEQUENCE', 'pir': b'>..;', 'qual': b'>', 'swiss': b'ID ', 'uniprot-xml': b'<entry '}[format]
        self._marker = marker
        self._marker_re = re.compile(b'^' + marker)

    def __iter__(self):
        """Return (id, offset, length) tuples."""
        marker_offset = len(self._marker)
        marker_re = self._marker_re
        handle = self._handle
        handle.seek(0)
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
        while marker_re.match(line):
            id = line[marker_offset:].strip().split(None, 1)[0]
            length = len(line)
            while True:
                end_offset = handle.tell()
                line = handle.readline()
                if marker_re.match(line) or not line:
                    yield (id.decode(), start_offset, length)
                    start_offset = end_offset
                    break
                else:
                    length += len(line)
        assert not line, repr(line)

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        marker_re = self._marker_re
        handle.seek(offset)
        lines = [handle.readline()]
        while True:
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
            lines.append(line)
        return b''.join(lines)