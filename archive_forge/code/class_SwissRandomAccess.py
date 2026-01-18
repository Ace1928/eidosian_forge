import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class SwissRandomAccess(SequentialSeqFileRandomAccess):
    """Random access to a SwissProt file."""

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        marker_re = self._marker_re
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
        while marker_re.match(line):
            length = len(line)
            line = handle.readline()
            length += len(line)
            assert line.startswith(b'AC ')
            key = line[3:].strip().split(b';')[0].strip()
            while True:
                end_offset = handle.tell()
                line = handle.readline()
                if marker_re.match(line) or not line:
                    yield (key.decode(), start_offset, length)
                    start_offset = end_offset
                    break
                length += len(line)
        assert not line, repr(line)