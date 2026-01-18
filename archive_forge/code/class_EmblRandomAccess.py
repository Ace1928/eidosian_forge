import re
from io import BytesIO
from io import StringIO
from Bio import SeqIO
from Bio.File import _IndexedSeqFileProxy
from Bio.File import _open_for_random_access
class EmblRandomAccess(SequentialSeqFileRandomAccess):
    """Indexed dictionary like access to an EMBL file."""

    def __iter__(self):
        """Iterate over the sequence records in the file."""
        handle = self._handle
        handle.seek(0)
        marker_re = self._marker_re
        sv_marker = b'SV '
        ac_marker = b'AC '
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if marker_re.match(line) or not line:
                break
        while marker_re.match(line):
            setbysv = False
            length = len(line)
            if line[2:].count(b';') in [5, 6]:
                parts = line[3:].rstrip().split(b';')
                if parts[1].strip().startswith(sv_marker):
                    key = parts[0].strip() + b'.' + parts[1].strip().split()[1]
                    setbysv = True
                else:
                    key = parts[0].strip()
            elif line[2:].count(b';') in [2, 3]:
                key = line[3:].strip().split(None, 1)[0]
                if key.endswith(b';'):
                    key = key[:-1]
            else:
                raise ValueError(f'Did not recognise the ID line layout:\n{line!r}')
            while True:
                line = handle.readline()
                if marker_re.match(line) or not line:
                    end_offset = handle.tell() - len(line)
                    yield (key.decode(), start_offset, length)
                    start_offset = end_offset
                    break
                elif line.startswith(ac_marker) and (not setbysv):
                    key = line.rstrip().split()[1]
                    if key.endswith(b';'):
                        key = key[:-1]
                elif line.startswith(sv_marker):
                    key = line.rstrip().split()[1]
                    setbysv = True
                length += len(line)
        assert not line, repr(line)