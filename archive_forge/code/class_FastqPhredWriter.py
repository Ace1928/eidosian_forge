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
class FastqPhredWriter(SequenceWriter):
    """Class to write standard FASTQ format files (using PHRED quality scores) (OBSOLETE).

    Although you can use this class directly, you are strongly encouraged
    to use the ``as_fastq`` function, or top level ``Bio.SeqIO.write()``
    function instead via the format name "fastq" or the alias "fastq-sanger".

    For example, this code reads in a standard Sanger style FASTQ file
    (using PHRED scores) and re-saves it as another Sanger style FASTQ file:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/example.fastq", "fastq")
    >>> with open("Quality/temp.fastq", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "fastq")
    3

    You might want to do this if the original file included extra line breaks,
    which while valid may not be supported by all tools.  The output file from
    Biopython will have each sequence on a single line, and each quality
    string on a single line (which is considered desirable for maximum
    compatibility).

    In this next example, an old style Solexa/Illumina FASTQ file (using Solexa
    quality scores) is converted into a standard Sanger style FASTQ file using
    PHRED qualities:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/solexa_example.fastq", "fastq-solexa")
    >>> with open("Quality/temp.fastq", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "fastq")
    5

    This code is also called if you use the .format("fastq") method of a
    SeqRecord, or .format("fastq-sanger") if you prefer that alias.

    Note that Sanger FASTQ files have an upper limit of PHRED quality 93, which is
    encoded as ASCII 126, the tilde. If your quality scores are truncated to fit, a
    warning is issued.

    P.S. To avoid cluttering up your working directory, you can delete this
    temporary file now:

    >>> import os
    >>> os.remove("Quality/temp.fastq")
    """

    def write_record(self, record: SeqRecord) -> None:
        """Write a single FASTQ record to the file."""
        self._record_written = True
        seq = record.seq
        if seq is None:
            raise ValueError(f'No sequence for record {record.id}')
        qualities_str = _get_sanger_quality_str(record)
        if len(qualities_str) != len(seq):
            raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq), len(qualities_str)))
        id_ = self.clean(record.id) if record.id else ''
        description = self.clean(record.description)
        if description and description.split(None, 1)[0] == id_:
            title = description
        elif description:
            title = f'{id_} {description}'
        else:
            title = id_
        self.handle.write(f'@{title}\n{seq}\n+\n{qualities_str}\n')