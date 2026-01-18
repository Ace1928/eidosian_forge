import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _read_chunk(self):
    """
        Read and decode the next chunk of data from the BufferedReader.
        """
    if self._decoder is None:
        raise ValueError('no decoder')
    if self._telling:
        dec_buffer, dec_flags = self._decoder.getstate()
    if self._has_read1:
        input_chunk = self.buffer.read1(self._CHUNK_SIZE)
    else:
        input_chunk = self.buffer.read(self._CHUNK_SIZE)
    eof = not input_chunk
    decoded_chars = self._decoder.decode(input_chunk, eof)
    self._set_decoded_chars(decoded_chars)
    if decoded_chars:
        self._b2cratio = len(input_chunk) / len(self._decoded_chars)
    else:
        self._b2cratio = 0.0
    if self._telling:
        self._snapshot = (dec_flags, dec_buffer + input_chunk)
    return not eof