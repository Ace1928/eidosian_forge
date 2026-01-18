import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, block_align, mmap=False):
    """
    Notes
    -----
    Assumes file pointer is immediately after the 'data' id

    It's possible to not use all available bits in a container, or to store
    samples in a container bigger than necessary, so bytes_per_sample uses
    the actual reported container size (nBlockAlign / nChannels).  Real-world
    examples:

    Adobe Audition's "24-bit packed int (type 1, 20-bit)"

        nChannels = 2, nBlockAlign = 6, wBitsPerSample = 20

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-int12-AFsp.wav
    is:

        nChannels = 2, nBlockAlign = 4, wBitsPerSample = 12

    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf
    gives an example of:

        nChannels = 2, nBlockAlign = 8, wBitsPerSample = 20
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'
    size = struct.unpack(fmt + 'I', fid.read(4))[0]
    bytes_per_sample = block_align // channels
    n_samples = size // bytes_per_sample
    if format_tag == WAVE_FORMAT.PCM:
        if 1 <= bit_depth <= 8:
            dtype = 'u1'
        elif bytes_per_sample in {3, 5, 6, 7}:
            dtype = 'V1'
        elif bit_depth <= 64:
            dtype = f'{fmt}i{bytes_per_sample}'
        else:
            raise ValueError(f'Unsupported bit depth: the WAV file has {bit_depth}-bit integer data.')
    elif format_tag == WAVE_FORMAT.IEEE_FLOAT:
        if bit_depth in {32, 64}:
            dtype = f'{fmt}f{bytes_per_sample}'
        else:
            raise ValueError(f'Unsupported bit depth: the WAV file has {bit_depth}-bit floating-point data.')
    else:
        _raise_bad_format(format_tag)
    start = fid.tell()
    if not mmap:
        try:
            count = size if dtype == 'V1' else n_samples
            data = numpy.fromfile(fid, dtype=dtype, count=count)
        except io.UnsupportedOperation:
            fid.seek(start, 0)
            data = numpy.frombuffer(fid.read(size), dtype=dtype)
        if dtype == 'V1':
            dt = f'{fmt}i4' if bytes_per_sample == 3 else f'{fmt}i8'
            a = numpy.zeros((len(data) // bytes_per_sample, numpy.dtype(dt).itemsize), dtype='V1')
            if is_big_endian:
                a[:, :bytes_per_sample] = data.reshape((-1, bytes_per_sample))
            else:
                a[:, -bytes_per_sample:] = data.reshape((-1, bytes_per_sample))
            data = a.view(dt).reshape(a.shape[:-1])
    elif bytes_per_sample in {1, 2, 4, 8}:
        start = fid.tell()
        data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start, shape=(n_samples,))
        fid.seek(start + size)
    else:
        raise ValueError(f'mmap=True not compatible with {bytes_per_sample}-byte container size.')
    _handle_pad_byte(fid, size)
    if channels > 1:
        data = data.reshape(-1, channels)
    return data