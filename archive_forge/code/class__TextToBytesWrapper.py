import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
class _TextToBytesWrapper(io.BufferedReader):
    """
    Convert a TextIOBase string stream to a byte stream.
    """

    def __init__(self, text_io_buffer, encoding=None, errors=None, **kwargs):
        super().__init__(text_io_buffer, **kwargs)
        self.encoding = encoding or text_io_buffer.encoding or 'utf-8'
        self.errors = errors or text_io_buffer.errors or 'strict'

    def __del__(self):
        self.detach()

    def _encoding_call(self, method_name, *args, **kwargs):
        raw_method = getattr(self.raw, method_name)
        val = raw_method(*args, **kwargs)
        return val.encode(self.encoding, errors=self.errors)

    def read(self, size=-1):
        return self._encoding_call('read', size)

    def read1(self, size=-1):
        return self._encoding_call('read1', size)

    def peek(self, size=-1):
        return self._encoding_call('peek', size)

    def seek(self, offset, whence=0):
        if offset == 0 and whence == 0 or (offset == 0 and whence == 2):
            super().seek(offset, whence)
        else:
            pass