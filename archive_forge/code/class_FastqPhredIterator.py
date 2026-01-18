import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
class FastqPhredIterator(SequenceIterator[str]):
    """Parser for FASTQ files."""

    def __init__(self, source: _TextIOSource, alphabet: None=None):
        """Iterate over FASTQ records as SeqRecord objects.

        Arguments:
         - source - input stream opened in text mode, or a path to a file
         - alphabet - optional alphabet, no longer used. Leave as None.

        For each sequence in a (Sanger style) FASTQ file there is a matching string
        encoding the PHRED qualities (integers between 0 and about 90) using ASCII
        values with an offset of 33.

        For example, consider a file containing three short reads::

            @EAS54_6_R1_2_1_413_324
            CCCTTCTTGTCTTCAGCGTTTCTCC
            +
            ;;3;;;;;;;;;;;;7;;;;;;;88
            @EAS54_6_R1_2_1_540_792
            TTGGCAGGCCAAGGCCGATGGATCA
            +
            ;;;;;;;;;;;7;;;;;-;;;3;83
            @EAS54_6_R1_2_1_443_348
            GTTGCTTCTGGCGTGGGTGGGGGGG
            +
            ;;;;;;;;;;;9;7;;.7;393333

        For each sequence (e.g. "CCCTTCTTGTCTTCAGCGTTTCTCC") there is a matching
        string encoding the PHRED qualities using a ASCII values with an offset of
        33 (e.g. ";;3;;;;;;;;;;;;7;;;;;;;88").

        Using this module directly you might run:

        >>> with open("Quality/example.fastq") as handle:
        ...     for record in FastqPhredIterator(handle):
        ...         print("%s %s" % (record.id, record.seq))
        EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC
        EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA
        EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG

        Typically however, you would call this via Bio.SeqIO instead with "fastq"
        (or "fastq-sanger") as the format:

        >>> from Bio import SeqIO
        >>> with open("Quality/example.fastq") as handle:
        ...     for record in SeqIO.parse(handle, "fastq"):
        ...         print("%s %s" % (record.id, record.seq))
        EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC
        EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA
        EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG

        If you want to look at the qualities, they are record in each record's
        per-letter-annotation dictionary as a simple list of integers:

        >>> print(record.letter_annotations["phred_quality"])
        [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]

        To modify the records returned by the parser, you can use a generator
        function. For example, to store the mean PHRED quality in the record
        description, use

        >>> from statistics import mean
        >>> def modify_records(records):
        ...     for record in records:
        ...         record.description = mean(record.letter_annotations['phred_quality'])
        ...         yield record
        ...
        >>> with open('Quality/example.fastq') as handle:
        ...     for record in modify_records(FastqPhredIterator(handle)):
        ...         print(record.id, record.description)
        ...
        EAS54_6_R1_2_1_413_324 25.28
        EAS54_6_R1_2_1_540_792 24.52
        EAS54_6_R1_2_1_443_348 23.4

        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        super().__init__(source, mode='t', fmt='Fastq')

    def parse(self, handle: IO[str]) -> Iterator[SeqRecord]:
        """Start parsing the file, and return a SeqRecord iterator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle: IO[str]) -> Iterator[SeqRecord]:
        """Parse the file and generate SeqRecord objects."""
        assert SANGER_SCORE_OFFSET == ord('!')
        q_mapping = {chr(letter): letter - SANGER_SCORE_OFFSET for letter in range(SANGER_SCORE_OFFSET, 94 + SANGER_SCORE_OFFSET)}
        for title_line, seq_string, quality_string in FastqGeneralIterator(handle):
            descr = title_line
            id = descr.split()[0]
            name = id
            record = SeqRecord(Seq(seq_string), id=id, name=name, description=descr)
            try:
                qualities = [q_mapping[letter2] for letter2 in quality_string]
            except KeyError:
                raise ValueError('Invalid character in quality string') from None
            dict.__setitem__(record._per_letter_annotations, 'phred_quality', qualities)
            yield record