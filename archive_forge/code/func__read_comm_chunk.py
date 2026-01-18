import struct
import builtins
import warnings
from collections import namedtuple
def _read_comm_chunk(self, chunk):
    self._nchannels = _read_short(chunk)
    self._nframes = _read_long(chunk)
    self._sampwidth = (_read_short(chunk) + 7) // 8
    self._framerate = int(_read_float(chunk))
    if self._sampwidth <= 0:
        raise Error('bad sample width')
    if self._nchannels <= 0:
        raise Error('bad # of channels')
    self._framesize = self._nchannels * self._sampwidth
    if self._aifc:
        kludge = 0
        if chunk.chunksize == 18:
            kludge = 1
            warnings.warn('Warning: bad COMM chunk size')
            chunk.chunksize = 23
        self._comptype = chunk.read(4)
        if kludge:
            length = ord(chunk.file.read(1))
            if length & 1 == 0:
                length = length + 1
            chunk.chunksize = chunk.chunksize + length
            chunk.file.seek(-1, 1)
        self._compname = _read_string(chunk)
        if self._comptype != b'NONE':
            if self._comptype == b'G722':
                self._convert = self._adpcm2lin
            elif self._comptype in (b'ulaw', b'ULAW'):
                self._convert = self._ulaw2lin
            elif self._comptype in (b'alaw', b'ALAW'):
                self._convert = self._alaw2lin
            elif self._comptype in (b'sowt', b'SOWT'):
                self._convert = self._sowt2lin
            else:
                raise Error('unsupported compression type')
            self._sampwidth = 2
    else:
        self._comptype = b'NONE'
        self._compname = b'not compressed'