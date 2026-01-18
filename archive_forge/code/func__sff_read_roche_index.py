import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _sff_read_roche_index(handle):
    """Read any existing Roche style read index provided in the SFF file (PRIVATE).

    Will use the handle seek/tell functions.

    This works on ".srt1.00" and ".mft1.00" style Roche SFF index blocks.

    Roche SFF indices use base 255 not 256, meaning we see bytes in range the
    range 0 to 254 only. This appears to be so that byte 0xFF (character 255)
    can be used as a marker character to separate entries (required if the
    read name lengths vary).

    Note that since only four bytes are used for the read offset, this is
    limited to 255^4 bytes (nearly 4GB). If you try to use the Roche sfffile
    tool to combine SFF files beyond this limit, they issue a warning and
    omit the index (and manifest).
    """
    number_of_reads, header_length, index_offset, index_length, xml_offset, xml_size, read_index_offset, read_index_size = _sff_find_roche_index(handle)
    handle.seek(read_index_offset)
    fmt = '>5B'
    for read in range(number_of_reads):
        data = handle.read(6)
        while True:
            more = handle.read(1)
            if not more:
                raise ValueError('Premature end of file!')
            data += more
            if more == _flag:
                break
        assert data[-1:] == _flag, data[-1:]
        name = data[:-6].decode()
        off4, off3, off2, off1, off0 = struct.unpack(fmt, data[-6:-1])
        offset = off0 + 255 * off1 + 65025 * off2 + 16581375 * off3
        if off4:
            raise ValueError('Expected a null terminator to the read name.')
        yield (name, offset)
    if handle.tell() != read_index_offset + read_index_size:
        raise ValueError('Problem with index length? %i vs %i' % (handle.tell(), read_index_offset + read_index_size))