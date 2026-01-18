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
class QualPhredIterator(SequenceIterator):
    """Parser for QUAL files with PHRED quality scores but no sequence."""

    def __init__(self, source: _TextIOSource, alphabet: None=None) -> None:
        """For QUAL files which include PHRED quality scores, but no sequence.

        For example, consider this short QUAL file::

            >EAS54_6_R1_2_1_413_324
            26 26 18 26 26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26
            26 26 26 23 23
            >EAS54_6_R1_2_1_540_792
            26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26 26 12 26 26
            26 18 26 23 18
            >EAS54_6_R1_2_1_443_348
            26 26 26 26 26 26 26 26 26 26 26 24 26 22 26 26 13 22 26 18
            24 18 18 18 18

        Using this module directly you might run:

        >>> with open("Quality/example.qual") as handle:
        ...     for record in QualPhredIterator(handle):
        ...         print("%s read of length %d" % (record.id, len(record.seq)))
        EAS54_6_R1_2_1_413_324 read of length 25
        EAS54_6_R1_2_1_540_792 read of length 25
        EAS54_6_R1_2_1_443_348 read of length 25

        Typically however, you would call this via Bio.SeqIO instead with "qual"
        as the format:

        >>> from Bio import SeqIO
        >>> with open("Quality/example.qual") as handle:
        ...     for record in SeqIO.parse(handle, "qual"):
        ...         print("%s read of length %d" % (record.id, len(record.seq)))
        EAS54_6_R1_2_1_413_324 read of length 25
        EAS54_6_R1_2_1_540_792 read of length 25
        EAS54_6_R1_2_1_443_348 read of length 25

        Only the sequence length is known, as the QUAL file does not contain
        the sequence string itself.

        The quality scores themselves are available as a list of integers
        in each record's per-letter-annotation:

        >>> print(record.letter_annotations["phred_quality"])
        [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]

        You can still slice one of these SeqRecord objects:

        >>> sub_record = record[5:10]
        >>> print("%s %s" % (sub_record.id, sub_record.letter_annotations["phred_quality"]))
        EAS54_6_R1_2_1_443_348 [26, 26, 26, 26, 26]

        As of Biopython 1.59, this parser will accept files with negatives quality
        scores but will replace them with the lowest possible PHRED score of zero.
        This will trigger a warning, previously it raised a ValueError exception.
        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        super().__init__(source, mode='t', fmt='QUAL')

    def parse(self, handle: IO) -> Iterator[SeqRecord]:
        """Start parsing the file, and return a SeqRecord iterator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle: IO) -> Iterator[SeqRecord]:
        """Parse the file and generate SeqRecord objects."""
        for line in handle:
            if line[0] == '>':
                break
        else:
            return
        while True:
            if line[0] != '>':
                raise ValueError("Records in Fasta files should start with '>' character")
            descr = line[1:].rstrip()
            id = descr.split()[0]
            name = id
            qualities: List[int] = []
            for line in handle:
                if line[0] == '>':
                    break
                qualities.extend((int(word) for word in line.split()))
            else:
                line = None
            if qualities and min(qualities) < 0:
                warnings.warn('Negative quality score %i found, substituting PHRED zero instead.' % min(qualities), BiopythonParserWarning)
                qualities = [max(0, q) for q in qualities]
            sequence = Seq(None, length=len(qualities))
            record = SeqRecord(sequence, id=id, name=name, description=descr)
            dict.__setitem__(record._per_letter_annotations, 'phred_quality', qualities)
            yield record
            if line is None:
                return
        raise ValueError('Unrecognised QUAL record format.')