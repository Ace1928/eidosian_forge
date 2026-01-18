import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _get_topology(self, record):
    """Set the topology to 'circular', 'linear' if defined (PRIVATE)."""
    max_topology_len = len('circular')
    topology = self._get_annotation_str(record, 'topology', default='')
    if topology and len(topology) <= max_topology_len:
        return topology.ljust(max_topology_len)
    else:
        return ' ' * max_topology_len