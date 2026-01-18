import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
class _PeekableReader:
    """lightweight stream wrapper that implements peek()"""

    def __init__(self, stream):
        self.stream = stream

    def read(self, n):
        return self.stream.read(n)

    def readline(self):
        return self.stream.readline()

    def tell(self):
        return self.stream.tell()

    def close(self):
        return self.stream.close()

    def peek(self, n):
        stream = self.stream
        try:
            if hasattr(stream, 'flush'):
                stream.flush()
            position = stream.tell()
            stream.seek(position)
            chunk = stream.read(n)
            stream.seek(position)
            return chunk
        except (AttributeError, OSError):
            raise NotImplementedError('stream is not peekable: %r', stream) from None