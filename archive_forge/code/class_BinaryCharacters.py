import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class BinaryCharacters(PhyloElement):
    """Binary characters at the root of a clade.

    The names and/or counts of binary characters present, gained, and lost
    at the root of a clade.
    """

    def __init__(self, type=None, gained_count=None, lost_count=None, present_count=None, absent_count=None, gained=None, lost=None, present=None, absent=None):
        """Initialize values for the BinaryCharacters object."""
        self.type = type
        self.gained_count = gained_count
        self.lost_count = lost_count
        self.present_count = present_count
        self.absent_count = absent_count
        self.gained = gained or []
        self.lost = lost or []
        self.present = present or []
        self.absent = absent or []