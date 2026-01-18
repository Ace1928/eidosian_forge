from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_interleaved_other_blocks(self, stream, seqs):
    i = 0
    for line in stream:
        line = line.rstrip()
        if not line:
            assert i == self._number_of_seqs
            i = 0
        else:
            seq = line.replace(' ', '')
            seqs[i].append(seq)
            i += 1
    if i != 0 and i != self._number_of_seqs:
        raise ValueError('Unexpected file format')