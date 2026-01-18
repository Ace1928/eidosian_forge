from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
def _is_header(self, line):
    line = line.strip()
    parts = [x for x in line.split() if x]
    if len(parts) != 2:
        return False
    try:
        number_of_seqs = int(parts[0])
        length_of_seqs = int(parts[1])
        return True
    except ValueError:
        return False