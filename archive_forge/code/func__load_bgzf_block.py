import struct
import sys
import zlib
from builtins import open as _open
def _load_bgzf_block(handle, text_mode=False):
    """Load the next BGZF block of compressed data (PRIVATE).

    Returns a tuple (block size and data), or at end of file
    will raise StopIteration.
    """
    magic = handle.read(4)
    if not magic:
        raise StopIteration
    if magic != _bgzf_magic:
        raise ValueError('A BGZF (e.g. a BAM file) block should start with %r, not %r; handle.tell() now says %r' % (_bgzf_magic, magic, handle.tell()))
    gzip_mod_time, gzip_extra_flags, gzip_os, extra_len = struct.unpack('<LBBH', handle.read(8))
    block_size = None
    x_len = 0
    while x_len < extra_len:
        subfield_id = handle.read(2)
        subfield_len = struct.unpack('<H', handle.read(2))[0]
        subfield_data = handle.read(subfield_len)
        x_len += subfield_len + 4
        if subfield_id == _bytes_BC:
            if subfield_len != 2:
                raise ValueError('Wrong BC payload length')
            if block_size is not None:
                raise ValueError('Two BC subfields?')
            block_size = struct.unpack('<H', subfield_data)[0] + 1
    if x_len != extra_len:
        raise ValueError(f'x_len and extra_len differ {x_len}, {extra_len}')
    if block_size is None:
        raise ValueError("Missing BC, this isn't a BGZF file!")
    deflate_size = block_size - 1 - extra_len - 19
    d = zlib.decompressobj(-15)
    data = d.decompress(handle.read(deflate_size)) + d.flush()
    expected_crc = handle.read(4)
    expected_size = struct.unpack('<I', handle.read(4))[0]
    if expected_size != len(data):
        raise RuntimeError('Decompressed to %i, not %i' % (len(data), expected_size))
    crc = zlib.crc32(data)
    if crc < 0:
        crc = struct.pack('<i', crc)
    else:
        crc = struct.pack('<I', crc)
    if expected_crc != crc:
        raise RuntimeError(f'CRC is {crc}, not {expected_crc}')
    if text_mode:
        return (block_size, data.decode('latin-1'))
    else:
        return (block_size, data)