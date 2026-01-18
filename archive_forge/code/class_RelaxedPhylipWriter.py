from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class RelaxedPhylipWriter(PhylipWriter):
    """Relaxed Phylip format writer."""

    def write_alignment(self, alignment):
        """Write a relaxed phylip alignment."""
        for name in (s.id.strip() for s in alignment):
            if any((c in name for c in string.whitespace)):
                raise ValueError(f'Whitespace not allowed in identifier: {name}')
        if len(alignment) == 0:
            id_width = 1
        else:
            id_width = max((len(s.id.strip()) for s in alignment)) + 1
        super().write_alignment(alignment, id_width)