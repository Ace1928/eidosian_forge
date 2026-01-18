from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class PhylipWriter(SequentialAlignmentWriter):
    """Phylip alignment writer."""

    def write_alignment(self, alignment, id_width=_PHYLIP_ID_WIDTH):
        """Use this to write (another) single alignment to an open file.

        This code will write interlaced alignments (when the sequences are
        longer than 50 characters).

        Note that record identifiers are strictly truncated to id_width,
        defaulting to the value required to comply with the PHYLIP standard.

        For more information on the file format, please see:
        http://evolution.genetics.washington.edu/phylip/doc/sequence.html
        http://evolution.genetics.washington.edu/phylip/doc/main.html#inputfiles
        """
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
        seqs = []
        for record in alignment:
            '\n            Quoting the PHYLIP version 3.6 documentation:\n\n            The name should be ten characters in length, filled out to\n            the full ten characters by blanks if shorter. Any printable\n            ASCII/ISO character is allowed in the name, except for\n            parentheses ("(" and ")"), square brackets ("[" and "]"),\n            colon (":"), semicolon (";") and comma (","). If you forget\n            to extend the names to ten characters in length by blanks,\n            the program [i.e. PHYLIP] will get out of synchronization\n            with the contents of the data file, and an error message will\n            result.\n\n            Note that Tab characters count as only one character in the\n            species names. Their inclusion can cause trouble.\n            '
            name = sanitize_name(record.id, id_width)
            if name in names:
                raise ValueError('Repeated name %r (originally %r), possibly due to truncation' % (name, record.id))
            names.append(name)
            sequence = str(record.seq)
            if '.' in sequence:
                raise ValueError(_NO_DOTS)
            seqs.append(sequence)
        handle.write(' %i %s\n' % (len(alignment), length_of_seqs))
        block = 0
        while True:
            for name, sequence in zip(names, seqs):
                if block == 0:
                    handle.write(name[:id_width].ljust(id_width))
                else:
                    handle.write(' ' * id_width)
                for chunk in range(5):
                    i = block * 50 + chunk * 10
                    seq_segment = sequence[i:i + 10]
                    handle.write(f' {seq_segment}')
                    if i + 10 > length_of_seqs:
                        break
                handle.write('\n')
            block += 1
            if block * 50 >= length_of_seqs:
                break
            handle.write('\n')