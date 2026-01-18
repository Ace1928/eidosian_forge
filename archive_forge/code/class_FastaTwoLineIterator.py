import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
class FastaTwoLineIterator(SequenceIterator):
    """Parser for Fasta files with exactly two lines per record."""

    def __init__(self, source):
        """Iterate over two-line Fasta records (as SeqRecord objects).

        Arguments:
         - source - input stream opened in text mode, or a path to a file

        This uses a strict interpretation of the FASTA as requiring
        exactly two lines per record (no line wrapping).

        Only the default title to ID/name/description parsing offered
        by the relaxed FASTA parser is offered.
        """
        super().__init__(source, mode='t', fmt='FASTA')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        for title, sequence in FastaTwoLineParser(handle):
            try:
                first_word = title.split(None, 1)[0]
            except IndexError:
                assert not title, repr(title)
                first_word = ''
            yield SeqRecord(Seq(sequence), id=first_word, name=first_word, description=title)