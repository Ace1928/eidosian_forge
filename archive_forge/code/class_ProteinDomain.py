import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class ProteinDomain(PhyloElement):
    """Represents an individual domain in a domain architecture.

    The locations use 0-based indexing, as most Python objects including
    SeqFeature do, rather than the usual biological convention starting at 1.
    This means the start and end attributes can be used directly as slice
    indexes on Seq objects.

    :Parameters:
        start : non-negative integer
            start of the domain on the sequence, using 0-based indexing
        end : non-negative integer
            end of the domain on the sequence
        confidence : float
            can be used to store e.g. E-values
        id : string
            unique identifier/name

    """

    def __init__(self, value, start, end, confidence=None, id=None):
        """Initialize value for a ProteinDomain object."""
        self.value = value
        self.start = start
        self.end = end
        self.confidence = confidence
        self.id = id

    @classmethod
    def from_seqfeature(cls, feat):
        """Create ProteinDomain object from SeqFeature."""
        return ProteinDomain(feat.id, feat.location.start, feat.location.end, confidence=feat.qualifiers.get('confidence'))

    def to_seqfeature(self):
        """Create a SeqFeature from the ProteinDomain Object."""
        feat = SeqFeature(location=SimpleLocation(self.start, self.end), id=self.value)
        try:
            confidence = self.confidence
        except AttributeError:
            pass
        else:
            feat.qualifiers['confidence'] = confidence
        return feat