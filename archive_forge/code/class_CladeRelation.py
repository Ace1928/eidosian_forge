import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class CladeRelation(PhyloElement):
    """Expresses a typed relationship between two clades.

    For example, this could be used to describe multiple parents of a clade.

    :type id_ref_0: str
    :type id_ref_1: str
    :type distance: str
    :type type: str

    :type confidence: Confidence
    """

    def __init__(self, type, id_ref_0, id_ref_1, distance=None, confidence=None):
        """Initialize values for the CladeRelation object."""
        self.distance = distance
        self.type = type
        self.id_ref_0 = id_ref_0
        self.id_ref_1 = id_ref_1
        self.confidence = confidence