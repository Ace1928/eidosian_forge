import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class SffIterator(SequenceIterator):
    """Parser for Standard Flowgram Format (SFF) files."""

    def __init__(self, source, alphabet=None, trim=False):
        """Iterate over Standard Flowgram Format (SFF) reads (as SeqRecord objects).

            - source - path to an SFF file, e.g. from Roche 454 sequencing,
              or a file-like object opened in binary mode.
            - alphabet - optional alphabet, unused. Leave as None.
            - trim - should the sequences be trimmed?

        The resulting SeqRecord objects should match those from a paired FASTA
        and QUAL file converted from the SFF file using the Roche 454 tool
        ssfinfo. i.e. The sequence will be mixed case, with the trim regions
        shown in lower case.

        This function is used internally via the Bio.SeqIO functions:

        >>> from Bio import SeqIO
        >>> for record in SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff"):
        ...     print("%s %i" % (record.id, len(record)))
        ...
        E3MFGYR02JWQ7T 265
        E3MFGYR02JA6IL 271
        E3MFGYR02JHD4H 310
        E3MFGYR02GFKUC 299
        E3MFGYR02FTGED 281
        E3MFGYR02FR9G7 261
        E3MFGYR02GAZMS 278
        E3MFGYR02HHZ8O 221
        E3MFGYR02GPGB1 269
        E3MFGYR02F7Z7G 219

        You can also call it directly:

        >>> with open("Roche/E3MFGYR02_random_10_reads.sff", "rb") as handle:
        ...     for record in SffIterator(handle):
        ...         print("%s %i" % (record.id, len(record)))
        ...
        E3MFGYR02JWQ7T 265
        E3MFGYR02JA6IL 271
        E3MFGYR02JHD4H 310
        E3MFGYR02GFKUC 299
        E3MFGYR02FTGED 281
        E3MFGYR02FR9G7 261
        E3MFGYR02GAZMS 278
        E3MFGYR02HHZ8O 221
        E3MFGYR02GPGB1 269
        E3MFGYR02F7Z7G 219

        Or, with the trim option:

        >>> with open("Roche/E3MFGYR02_random_10_reads.sff", "rb") as handle:
        ...     for record in SffIterator(handle, trim=True):
        ...         print("%s %i" % (record.id, len(record)))
        ...
        E3MFGYR02JWQ7T 260
        E3MFGYR02JA6IL 265
        E3MFGYR02JHD4H 292
        E3MFGYR02GFKUC 295
        E3MFGYR02FTGED 277
        E3MFGYR02FR9G7 256
        E3MFGYR02GAZMS 271
        E3MFGYR02HHZ8O 150
        E3MFGYR02GPGB1 221
        E3MFGYR02F7Z7G 130

        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        super().__init__(source, mode='b', fmt='SFF')
        self.trim = trim

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        try:
            if 0 != handle.tell():
                raise ValueError('Not at start of file, offset %i' % handle.tell())
        except AttributeError:
            handle = _AddTellHandle(handle)
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Parse the file and generate SeqRecord objects."""
        trim = self.trim
        header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence = _sff_file_header(handle)
        read_header_fmt = '>2HI4H'
        read_header_size = struct.calcsize(read_header_fmt)
        read_flow_fmt = '>%iH' % number_of_flows_per_read
        read_flow_size = struct.calcsize(read_flow_fmt)
        assert 1 == struct.calcsize('>B')
        assert 1 == struct.calcsize('>s')
        assert 1 == struct.calcsize('>c')
        assert read_header_size % 8 == 0
        for read in range(number_of_reads):
            if index_offset and handle.tell() == index_offset:
                offset = index_offset + index_length
                if offset % 8:
                    offset += 8 - offset % 8
                assert offset % 8 == 0
                handle.seek(offset)
                index_offset = 0
            yield _sff_read_seq_record(handle, number_of_flows_per_read, flow_chars, key_sequence, trim)
        _check_eof(handle, index_offset, index_length)