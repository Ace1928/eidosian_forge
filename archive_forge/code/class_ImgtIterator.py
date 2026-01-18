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
class ImgtIterator(SequenceIterator):
    """Parser for IMGT files."""

    def __init__(self, source):
        """Break up an IMGT file into SeqRecord objects.

        Argument source is a file-like object opened in text mode or a path to a file.
        Every section from the LOCUS line to the terminating // becomes
        a single SeqRecord with associated annotation and features.

        Note that for genomes or chromosomes, there is typically only
        one record.
        """
        super().__init__(source, mode='t', fmt='IMGT')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = _ImgtScanner(debug=0).parse_records(handle)
        return records