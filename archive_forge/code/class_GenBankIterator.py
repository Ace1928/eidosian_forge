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
class GenBankIterator(SequenceIterator):
    """Parser for GenBank files."""

    def __init__(self, source):
        """Break up a Genbank file into SeqRecord objects.

        Argument source is a file-like object opened in text mode or a path to a file.
        Every section from the LOCUS line to the terminating // becomes
        a single SeqRecord with associated annotation and features.

        Note that for genomes or chromosomes, there is typically only
        one record.

        This gets called internally by Bio.SeqIO for the GenBank file format:

        >>> from Bio import SeqIO
        >>> for record in SeqIO.parse("GenBank/cor6_6.gb", "gb"):
        ...     print(record.id)
        ...
        X55053.1
        X62281.1
        M81224.1
        AJ237582.1
        L31939.1
        AF297471.1

        Equivalently,

        >>> with open("GenBank/cor6_6.gb") as handle:
        ...     for record in GenBankIterator(handle):
        ...         print(record.id)
        ...
        X55053.1
        X62281.1
        M81224.1
        AJ237582.1
        L31939.1
        AF297471.1

        """
        super().__init__(source, mode='t', fmt='GenBank')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = GenBankScanner(debug=0).parse_records(handle)
        return records