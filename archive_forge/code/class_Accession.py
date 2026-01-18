import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Accession(PhyloElement):
    """Captures the local part in a sequence identifier.

    Example: In ``UniProtKB:P17304``, the Accession instance attribute ``value``
    is 'P17304' and the ``source`` attribute is 'UniProtKB'.
    """

    def __init__(self, value, source):
        """Initialize value for Accession object."""
        self.value = value
        self.source = source

    def __str__(self):
        """Show the class name and an identifying attribute."""
        return f'{self.source}:{self.value}'