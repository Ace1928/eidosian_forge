import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Uri(PhyloElement):
    """A uniform resource identifier.

    In general, this is expected to be an URL (for example, to link to an image
    on a website, in which case the ``type`` attribute might be 'image' and
    ``desc`` might be 'image of a California sea hare').
    """

    def __init__(self, value, desc=None, type=None):
        """Initialize the class."""
        self.value = value
        self.desc = desc
        self.type = type

    def __str__(self):
        """Return string representation of Uri."""
        if self.value:
            return self.value
        return repr(self)