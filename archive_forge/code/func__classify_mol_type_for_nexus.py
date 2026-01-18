from io import StringIO
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.SeqRecord import SeqRecord
from Bio.Nexus import Nexus
def _classify_mol_type_for_nexus(self, alignment):
    """Return 'protein', 'dna', or 'rna' based on records' molecule type (PRIVATE).

        All the records must have a molecule_type annotation, and they must
        agree.

        Raises an exception if this is not possible.
        """
    values = {sequence.annotations.get('molecule_type', None) for sequence in alignment.sequences}
    if all((_ and 'DNA' in _ for _ in values)):
        return 'dna'
    elif all((_ and 'RNA' in _ for _ in values)):
        return 'rna'
    elif all((_ and 'protein' in _ for _ in values)):
        return 'protein'
    else:
        raise ValueError('Need the molecule type to be defined')