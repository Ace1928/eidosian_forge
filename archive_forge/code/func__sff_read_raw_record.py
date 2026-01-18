import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _sff_read_raw_record(handle, number_of_flows_per_read):
    """Extract the next read in the file as a raw (bytes) string (PRIVATE)."""
    read_header_fmt = '>2HI'
    read_header_size = struct.calcsize(read_header_fmt)
    read_flow_fmt = '>%iH' % number_of_flows_per_read
    read_flow_size = struct.calcsize(read_flow_fmt)
    raw = handle.read(read_header_size)
    read_header_length, name_length, seq_len = struct.unpack(read_header_fmt, raw)
    if read_header_length < 10 or read_header_length % 8 != 0:
        raise ValueError('Malformed read header, says length is %i' % read_header_length)
    raw += handle.read(8 + name_length)
    padding = read_header_length - read_header_size - 8 - name_length
    pad = handle.read(padding)
    if pad.count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post name %i byte padding region contained data' % padding, BiopythonParserWarning)
    raw += pad
    raw += handle.read(read_flow_size + seq_len * 3)
    padding = (read_flow_size + seq_len * 3) % 8
    if padding:
        padding = 8 - padding
        pad = handle.read(padding)
        if pad.count(_null) != padding:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Your SFF file is invalid, post quality %i byte padding region contained data' % padding, BiopythonParserWarning)
        raw += pad
    return raw