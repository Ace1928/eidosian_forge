from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class SequentialPhylipWriter(SequentialAlignmentWriter):
    """Sequential Phylip format Writer."""

    def write_alignment(self, alignment, id_width=_PHYLIP_ID_WIDTH):
        """Write a Phylip alignment to the handle."""
        handle = self.handle
        if len(alignment) == 0:
            raise ValueError('Must have at least one sequence')
        length_of_seqs = alignment.get_alignment_length()
        for record in alignment:
            if length_of_seqs != len(record.seq):
                raise ValueError('Sequences must all be the same length')
        if length_of_seqs <= 0:
            raise ValueError('Non-empty sequences are required')
        names = []
        for record in alignment:
            name = sanitize_name(record.id, id_width)
            if name in names:
                raise ValueError('Repeated name %r (originally %r), possibly due to truncation' % (name, record.id))
            names.append(name)
        handle.write(' %i %s\n' % (len(alignment), length_of_seqs))
        for name, record in zip(names, alignment):
            sequence = str(record.seq)
            if '.' in sequence:
                raise ValueError(_NO_DOTS)
            handle.write(name[:id_width].ljust(id_width))
            handle.write(sequence)
            handle.write('\n')