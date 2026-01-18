import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class MolSeq(PhyloElement):
    """Store a molecular sequence.

    :Parameters:
        value : string
            the sequence itself
        is_aligned : bool
            True if this sequence is aligned with the others (usually meaning
            all aligned seqs are the same length and gaps may be present)

    """
    re_value = re.compile('[a-zA-Z\\.\\-\\?\\*_]+')

    def __init__(self, value, is_aligned=None):
        """Initialize parameters for the MolSeq object."""
        _check_str(value, self.re_value.match)
        self.value = value
        self.is_aligned = is_aligned

    def __str__(self):
        """Return the value of the Molecular Sequence object."""
        return self.value