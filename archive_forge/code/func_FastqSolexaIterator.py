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
def FastqSolexaIterator(source: _TextIOSource, alphabet: None=None) -> Iterator[SeqRecord]:
    """Parse old Solexa/Illumina FASTQ like files (which differ in the quality mapping).

    The optional arguments are the same as those for the FastqPhredIterator.

    For each sequence in Solexa/Illumina FASTQ files there is a matching string
    encoding the Solexa integer qualities using ASCII values with an offset
    of 64.  Solexa scores are scaled differently to PHRED scores, and Biopython
    will NOT perform any automatic conversion when loading.

    NOTE - This file format is used by the OLD versions of the Solexa/Illumina
    pipeline. See also the FastqIlluminaIterator function for the NEW version.

    For example, consider a file containing these five records::

        @SLXA-B3_649_FC8437_R1_1_1_610_79
        GATGTGCAATACCTTTGTAGAGGAA
        +SLXA-B3_649_FC8437_R1_1_1_610_79
        YYYYYYYYYYYYYYYYYYWYWYYSU
        @SLXA-B3_649_FC8437_R1_1_1_397_389
        GGTTTGAGAAAGAGAAATGAGATAA
        +SLXA-B3_649_FC8437_R1_1_1_397_389
        YYYYYYYYYWYYYYWWYYYWYWYWW
        @SLXA-B3_649_FC8437_R1_1_1_850_123
        GAGGGTGTTGATCATGATGATGGCG
        +SLXA-B3_649_FC8437_R1_1_1_850_123
        YYYYYYYYYYYYYWYYWYYSYYYSY
        @SLXA-B3_649_FC8437_R1_1_1_362_549
        GGAAACAAAGTTTTTCTCAACATAG
        +SLXA-B3_649_FC8437_R1_1_1_362_549
        YYYYYYYYYYYYYYYYYYWWWWYWY
        @SLXA-B3_649_FC8437_R1_1_1_183_714
        GTATTATTTAATGGCATACACTCAA
        +SLXA-B3_649_FC8437_R1_1_1_183_714
        YYYYYYYYYYWYYYYWYWWUWWWQQ

    Using this module directly you might run:

    >>> with open("Quality/solexa_example.fastq") as handle:
    ...     for record in FastqSolexaIterator(handle):
    ...         print("%s %s" % (record.id, record.seq))
    SLXA-B3_649_FC8437_R1_1_1_610_79 GATGTGCAATACCTTTGTAGAGGAA
    SLXA-B3_649_FC8437_R1_1_1_397_389 GGTTTGAGAAAGAGAAATGAGATAA
    SLXA-B3_649_FC8437_R1_1_1_850_123 GAGGGTGTTGATCATGATGATGGCG
    SLXA-B3_649_FC8437_R1_1_1_362_549 GGAAACAAAGTTTTTCTCAACATAG
    SLXA-B3_649_FC8437_R1_1_1_183_714 GTATTATTTAATGGCATACACTCAA

    Typically however, you would call this via Bio.SeqIO instead with
    "fastq-solexa" as the format:

    >>> from Bio import SeqIO
    >>> with open("Quality/solexa_example.fastq") as handle:
    ...     for record in SeqIO.parse(handle, "fastq-solexa"):
    ...         print("%s %s" % (record.id, record.seq))
    SLXA-B3_649_FC8437_R1_1_1_610_79 GATGTGCAATACCTTTGTAGAGGAA
    SLXA-B3_649_FC8437_R1_1_1_397_389 GGTTTGAGAAAGAGAAATGAGATAA
    SLXA-B3_649_FC8437_R1_1_1_850_123 GAGGGTGTTGATCATGATGATGGCG
    SLXA-B3_649_FC8437_R1_1_1_362_549 GGAAACAAAGTTTTTCTCAACATAG
    SLXA-B3_649_FC8437_R1_1_1_183_714 GTATTATTTAATGGCATACACTCAA

    If you want to look at the qualities, they are recorded in each record's
    per-letter-annotation dictionary as a simple list of integers:

    >>> print(record.letter_annotations["solexa_quality"])
    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 23, 25, 25, 25, 25, 23, 25, 23, 23, 21, 23, 23, 23, 17, 17]

    These scores aren't very good, but they are high enough that they map
    almost exactly onto PHRED scores:

    >>> print("%0.2f" % phred_quality_from_solexa(25))
    25.01

    Let's look at faked example read which is even worse, where there are
    more noticeable differences between the Solexa and PHRED scores::

         @slxa_0001_1_0001_01
         ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN
         +slxa_0001_1_0001_01
         hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@?>=<;

    Again, you would typically use Bio.SeqIO to read this file in (rather than
    calling the Bio.SeqIO.QualtityIO module directly).  Most FASTQ files will
    contain thousands of reads, so you would normally use Bio.SeqIO.parse()
    as shown above.  This example has only as one entry, so instead we can
    use the Bio.SeqIO.read() function:

    >>> from Bio import SeqIO
    >>> with open("Quality/solexa_faked.fastq") as handle:
    ...     record = SeqIO.read(handle, "fastq-solexa")
    >>> print("%s %s" % (record.id, record.seq))
    slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN
    >>> print(record.letter_annotations["solexa_quality"])
    [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]

    These quality scores are so low that when converted from the Solexa scheme
    into PHRED scores they look quite different:

    >>> print("%0.2f" % phred_quality_from_solexa(-1))
    2.54
    >>> print("%0.2f" % phred_quality_from_solexa(-5))
    1.19

    Note you can use the Bio.SeqIO.write() function or the SeqRecord's format
    method to output the record(s):

    >>> print(record.format("fastq-solexa"))
    @slxa_0001_1_0001_01
    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN
    +
    hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@?>=<;
    <BLANKLINE>

    Note this output is slightly different from the input file as Biopython
    has left out the optional repetition of the sequence identifier on the "+"
    line.  If you want the to use PHRED scores, use "fastq" or "qual" as the
    output format instead, and Biopython will do the conversion for you:

    >>> print(record.format("fastq"))
    @slxa_0001_1_0001_01
    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN
    +
    IHGFEDCBA@?>=<;:9876543210/.-,++*)('&&%%$$##""
    <BLANKLINE>

    >>> print(record.format("qual"))
    >slxa_0001_1_0001_01
    40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21
    20 19 18 17 16 15 14 13 12 11 10 10 9 8 7 6 5 5 4 4 3 3 2 2
    1 1
    <BLANKLINE>

    As shown above, the poor quality Solexa reads have been mapped to the
    equivalent PHRED score (e.g. -5 to 1 as shown earlier).
    """
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    q_mapping = {chr(letter): letter - SOLEXA_SCORE_OFFSET for letter in range(SOLEXA_SCORE_OFFSET - 5, 63 + SOLEXA_SCORE_OFFSET)}
    for title_line, seq_string, quality_string in FastqGeneralIterator(source):
        descr = title_line
        id = descr.split()[0]
        name = id
        record = SeqRecord(Seq(seq_string), id=id, name=name, description=descr)
        try:
            qualities = [q_mapping[letter2] for letter2 in quality_string]
        except KeyError:
            raise ValueError('Invalid character in quality string') from None
        dict.__setitem__(record._per_letter_annotations, 'solexa_quality', qualities)
        yield record