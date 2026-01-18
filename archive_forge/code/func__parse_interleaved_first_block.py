from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_interleaved_first_block(self, lines, seqs, names):
    for line in lines:
        line = line.rstrip()
        name = line[:_PHYLIP_ID_WIDTH].strip()
        seq = line[_PHYLIP_ID_WIDTH:].strip().replace(' ', '')
        names.append(name)
        seqs.append([seq])