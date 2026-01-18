import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
class FastaIterator(SequenceIterator):
    """Parser for Fasta files."""

    def __init__(self, source: _TextIOSource, alphabet: None=None) -> None:
        """Iterate over Fasta records as SeqRecord objects.

        Arguments:
         - source - input stream opened in text mode, or a path to a file
         - alphabet - optional alphabet, not used. Leave as None.

        By default this will act like calling Bio.SeqIO.parse(handle, "fasta")
        with no custom handling of the title lines:

        >>> with open("Fasta/dups.fasta") as handle:
        ...     for record in FastaIterator(handle):
        ...         print(record.id)
        ...
        alpha
        beta
        gamma
        alpha
        delta

        If you want to modify the records before writing, for example to change
        the ID of each record, you can use a generator function as follows:

        >>> def modify_records(records):
        ...     for record in records:
        ...         record.id = record.id.upper()
        ...         yield record
        ...
        >>> with open('Fasta/dups.fasta') as handle:
        ...     for record in modify_records(FastaIterator(handle)):
        ...         print(record.id)
        ...
        ALPHA
        BETA
        GAMMA
        ALPHA
        DELTA

        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        super().__init__(source, mode='t', fmt='Fasta')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        for title, sequence in SimpleFastaParser(handle):
            try:
                first_word = title.split(None, 1)[0]
            except IndexError:
                assert not title, repr(title)
                first_word = ''
            yield SeqRecord(Seq(sequence), id=first_word, name=first_word, description=title)