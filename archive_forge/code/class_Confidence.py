import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Confidence(float, PhyloElement):
    """A general purpose confidence element.

    For example, this can be used to express the bootstrap support value of a
    clade (in which case the ``type`` attribute is 'bootstrap').

    :Parameters:
        value : float
            confidence value
        type : string
            label for the type of confidence, e.g. 'bootstrap'

    """

    def __new__(cls, value, type='unknown'):
        """Create and return a Confidence object with the specified value and type."""
        obj = super(Confidence, cls).__new__(cls, value)
        obj.type = type
        return obj

    @property
    def value(self):
        """Return the float value of the Confidence object."""
        return float(self)