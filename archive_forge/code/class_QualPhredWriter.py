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
class QualPhredWriter(SequenceWriter):
    """Class to write QUAL format files (using PHRED quality scores) (OBSOLETE).

    Although you can use this class directly, you are strongly encouraged
    to use the ``as_qual`` function, or top level ``Bio.SeqIO.write()``
    function instead.

    For example, this code reads in a FASTQ file and saves the quality scores
    into a QUAL file:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/example.fastq", "fastq")
    >>> with open("Quality/temp.qual", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "qual")
    3

    This code is also called if you use the .format("qual") method of a
    SeqRecord.

    P.S. Don't forget to clean up the temp file if you don't need it anymore:

    >>> import os
    >>> os.remove("Quality/temp.qual")
    """

    def __init__(self, handle: _TextIOSource, wrap: int=60, record2title: Optional[Callable[[SeqRecord], str]]=None) -> None:
        """Create a QUAL writer.

        Arguments:
         - handle - Handle to an output file, e.g. as returned
           by open(filename, "w")
         - wrap   - Optional line length used to wrap sequence lines.
           Defaults to wrapping the sequence at 60 characters. Use
           zero (or None) for no wrapping, giving a single long line
           for the sequence.
         - record2title - Optional function to return the text to be
           used for the title line of each record.  By default a
           combination of the record.id and record.description is
           used.  If the record.description starts with the record.id,
           then just the record.description is used.

        The record2title argument is present for consistency with the
        Bio.SeqIO.FastaIO writer class.
        """
        super().__init__(handle)
        self.wrap: Optional[int] = None
        if wrap:
            if wrap < 1:
                raise ValueError
            self.wrap = wrap
        self.record2title = record2title

    def write_record(self, record: SeqRecord) -> None:
        """Write a single QUAL record to the file."""
        self._record_written = True
        handle = self.handle
        wrap = self.wrap
        if self.record2title:
            title = self.clean(self.record2title(record))
        else:
            id_ = self.clean(record.id) if record.id else ''
            description = self.clean(record.description)
            if description and description.split(None, 1)[0] == id_:
                title = description
            elif description:
                title = f'{id} {description}'
            else:
                title = id_
        handle.write(f'>{title}\n')
        qualities = _get_phred_quality(record)
        try:
            qualities_strs = ['%i' % round(q, 0) for q in qualities]
        except TypeError:
            if None in qualities:
                raise TypeError('A quality value of None was found') from None
            else:
                raise
        if wrap is not None and wrap > 5:
            data = ' '.join(qualities_strs)
            while True:
                if len(data) <= wrap:
                    self.handle.write(data + '\n')
                    break
                else:
                    i = data.rfind(' ', 0, wrap)
                    handle.write(data[:i] + '\n')
                    data = data[i + 1:]
        elif wrap:
            while qualities_strs:
                line = qualities_strs.pop(0)
                while qualities_strs and len(line) + 1 + len(qualities_strs[0]) < wrap:
                    line += ' ' + qualities_strs.pop(0)
                handle.write(line + '\n')
        else:
            data = ' '.join(qualities_strs)
            handle.write(data + '\n')