import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def __maf_indexer(self):
    """Return index information for each bundle (PRIVATE).

        Yields index information for each bundle in the form of
        (bin, start, end, offset) tuples where start and end are
        0-based inclusive coordinates.
        """
    line = self._maf_fp.readline()
    while line:
        if line.startswith('a'):
            offset = self._maf_fp.tell() - len(line)
            while True:
                line = self._maf_fp.readline()
                if not line.strip() or line.startswith('a'):
                    raise ValueError('Target for indexing (%s) not found in this bundle' % (self._target_seqname,))
                elif line.startswith('s'):
                    line_split = line.strip().split()
                    if line_split[1] == self._target_seqname:
                        start = int(line_split[2])
                        size = int(line_split[3])
                        if size != len(line_split[6].replace('-', '')):
                            raise ValueError('Invalid length for target coordinates (expected %s, found %s)' % (size, len(line_split[6].replace('-', ''))))
                        end = start + size - 1
                        yield (self._ucscbin(start, end + 1), start, end, offset)
                        break
        line = self._maf_fp.readline()