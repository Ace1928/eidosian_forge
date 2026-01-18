import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _read_fmt_chunk(fid, is_big_endian):
    """
    Returns
    -------
    size : int
        size of format subchunk in bytes (minus 8 for "fmt " and itself)
    format_tag : int
        PCM, float, or compressed format
    channels : int
        number of channels
    fs : int
        sampling frequency in samples per second
    bytes_per_second : int
        overall byte rate for the file
    block_align : int
        bytes per sample, including all channels
    bit_depth : int
        bits per sample

    Notes
    -----
    Assumes file pointer is immediately after the 'fmt ' id
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'
    size = struct.unpack(fmt + 'I', fid.read(4))[0]
    if size < 16:
        raise ValueError('Binary structure of wave file is not compliant')
    res = struct.unpack(fmt + 'HHIIHH', fid.read(16))
    bytes_read = 16
    format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res
    if format_tag == WAVE_FORMAT.EXTENSIBLE and size >= 16 + 2:
        ext_chunk_size = struct.unpack(fmt + 'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2 + 4:2 + 4 + 16]
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xaa\x008\x9bq'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt + 'I', raw_guid[:4])[0]
        else:
            raise ValueError('Binary structure of wave file is not compliant')
    if format_tag not in KNOWN_WAVE_FORMATS:
        _raise_bad_format(format_tag)
    if size > bytes_read:
        fid.read(size - bytes_read)
    _handle_pad_byte(fid, size)
    if format_tag == WAVE_FORMAT.PCM:
        if bytes_per_second != fs * block_align:
            raise ValueError(f'WAV header is invalid: nAvgBytesPerSec must equal product of nSamplesPerSec and nBlockAlign, but file has nSamplesPerSec = {fs}, nBlockAlign = {block_align}, and nAvgBytesPerSec = {bytes_per_second}')
    return (size, format_tag, channels, fs, bytes_per_second, block_align, bit_depth)