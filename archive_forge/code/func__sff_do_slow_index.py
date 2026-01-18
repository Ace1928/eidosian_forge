import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _sff_do_slow_index(handle):
    """Generate an index by scanning though all the reads in an SFF file (PRIVATE).

    This is a slow but generic approach if we can't parse the provided index
    (if present).

    Will use the handle seek/tell functions.
    """
    handle.seek(0)
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
        record_offset = handle.tell()
        if record_offset == index_offset:
            offset = index_offset + index_length
            if offset % 8:
                offset += 8 - offset % 8
            assert offset % 8 == 0
            handle.seek(offset)
            record_offset = offset
        data = handle.read(read_header_size)
        read_header_length, name_length, seq_len, clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right = struct.unpack(read_header_fmt, data)
        if read_header_length < 10 or read_header_length % 8 != 0:
            raise ValueError('Malformed read header, says length is %i:\n%r' % (read_header_length, data))
        name = handle.read(name_length).decode()
        padding = read_header_length - read_header_size - name_length
        if handle.read(padding).count(_null) != padding:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Your SFF file is invalid, post name %i byte padding region contained data' % padding, BiopythonParserWarning)
        assert record_offset + read_header_length == handle.tell()
        size = read_flow_size + 3 * seq_len
        handle.seek(size, 1)
        padding = size % 8
        if padding:
            padding = 8 - padding
            if handle.read(padding).count(_null) != padding:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn('Your SFF file is invalid, post quality %i byte padding region contained data' % padding, BiopythonParserWarning)
        yield (name, record_offset)
    if handle.tell() % 8 != 0:
        raise ValueError('After scanning reads, did not end on a multiple of 8')