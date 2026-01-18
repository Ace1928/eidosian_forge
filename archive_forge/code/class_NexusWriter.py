from typing import IO, Iterator, Optional
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentWriter
from Bio.Nexus import Nexus
from Bio.SeqRecord import SeqRecord
class NexusWriter(AlignmentWriter):
    """Nexus alignment writer.

    Note that Nexus files are only expected to hold ONE alignment
    matrix.

    You are expected to call this class via the Bio.AlignIO.write() or
    Bio.SeqIO.write() functions.
    """

    def write_file(self, alignments):
        """Use this to write an entire file containing the given alignments.

        Arguments:
         - alignments - A list or iterator returning MultipleSeqAlignment objects.
           This should hold ONE and only one alignment.

        """
        align_iter = iter(alignments)
        try:
            alignment = next(align_iter)
        except StopIteration:
            return 0
        try:
            next(align_iter)
            raise ValueError('We can only write one Alignment to a Nexus file.')
        except StopIteration:
            pass
        self.write_alignment(alignment)
        return 1

    def write_alignment(self, alignment, interleave=None):
        """Write an alignment to file.

        Creates an empty Nexus object, adds the sequences
        and then gets Nexus to prepare the output.
        Default interleave behaviour: Interleave if columns > 1000
        --> Override with interleave=[True/False]
        """
        if len(alignment) == 0:
            raise ValueError('Must have at least one sequence')
        columns = alignment.get_alignment_length()
        if columns == 0:
            raise ValueError('Non-empty sequences are required')
        datatype = self._classify_mol_type_for_nexus(alignment)
        minimal_record = '#NEXUS\nbegin data; dimensions ntax=0 nchar=0; format datatype=%s; end;' % datatype
        n = Nexus.Nexus(minimal_record)
        for record in alignment:
            if datatype == 'dna' and 'U' in record.seq:
                raise ValueError(f'{record.id} contains U, but DNA alignment')
            elif datatype == 'rna' and 'T' in record.seq:
                raise ValueError(f'{record.id} contains T, but RNA alignment')
            n.add_sequence(record.id, str(record.seq))
        if interleave is None:
            interleave = columns > 1000
        n.write_nexus_data(self.handle, interleave=interleave)

    def _classify_mol_type_for_nexus(self, alignment):
        """Return 'protein', 'dna', or 'rna' based on records' molecule type (PRIVATE).

        All the records must have a molecule_type annotation, and they must
        agree.

        Raises an exception if this is not possible.
        """
        values = {_.annotations.get('molecule_type', None) for _ in alignment}
        if all((_ and 'DNA' in _ for _ in values)):
            return 'dna'
        elif all((_ and 'RNA' in _ for _ in values)):
            return 'rna'
        elif all((_ and 'protein' in _ for _ in values)):
            return 'protein'
        else:
            raise ValueError('Need the molecule type to be defined')