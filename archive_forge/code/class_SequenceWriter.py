from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class SequenceWriter:
    """Base class for sequence writers. This class should be subclassed.

    It is intended for sequential file formats with an (optional)
    header, repeated records, and an (optional) footer, as well
    as for interlaced file formats such as Clustal.

    The user may call the write_file() method to write a complete
    file containing the sequences.

    Alternatively, users may call the write_header(), followed
    by multiple calls to write_record() and/or write_records(),
    followed finally by write_footer().

    Note that write_header() cannot require any assumptions about
    the number of records.
    """

    def __init__(self, target: _IOSource, mode: str='w') -> None:
        """Create the writer object."""
        if mode == 'w':
            if isinstance(target, _PathLikeTypes):
                handle = open(target, mode)
            else:
                try:
                    handle = target
                    target.write('')
                except TypeError:
                    raise StreamModeError('File must be opened in text mode.') from None
        elif mode == 'wb':
            if isinstance(target, _PathLikeTypes):
                handle = open(target, mode)
            else:
                handle = target
                try:
                    target.write(b'')
                except TypeError:
                    raise StreamModeError('File must be opened in binary mode.') from None
        else:
            raise RuntimeError(f"Unknown mode '{mode}'")
        self._target = target
        self.handle = handle

    def clean(self, text: str) -> str:
        """Use this to avoid getting newlines in the output."""
        return text.replace('\n', ' ').replace('\r', ' ')

    def write_header(self):
        """Write the file header to the output file."""

    def write_footer(self):
        """Write the file footer to the output file."""

    def write_record(self, record):
        """Write a single record to the output file.

        record - a SeqRecord object
        """
        raise NotImplementedError('This method should be implemented')

    def write_records(self, records, maxcount=None):
        """Write records to the output file, and return the number of records.

        records - A list or iterator returning SeqRecord objects
        maxcount - The maximum number of records allowed by the
        file format, or None if there is no maximum.
        """
        count = 0
        if maxcount is None:
            for record in records:
                self.write_record(record)
                count += 1
        else:
            for record in records:
                if count == maxcount:
                    if maxcount == 1:
                        raise ValueError('More than one sequence found')
                    else:
                        raise ValueError('Number of sequences is larger than %d' % maxcount)
                self.write_record(record)
                count += 1
        return count

    def write_file(self, records, mincount=0, maxcount=None):
        """Write a complete file with the records, and return the number of records.

        records - A list or iterator returning SeqRecord objects
        """
        try:
            self.write_header()
            count = self.write_records(records, maxcount)
            self.write_footer()
        finally:
            if self.handle is not self._target:
                self.handle.close()
        if count < mincount:
            if mincount == 1:
                raise ValueError('Must have one sequence')
            elif mincount == maxcount:
                raise ValueError('Number of sequences is %d (expected %d)' % (count, mincount))
            else:
                raise ValueError('Number of sequences is %d (expected at least %d)' % (count, mincount))
        return count