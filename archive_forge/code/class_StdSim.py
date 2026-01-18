import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class StdSim:
    """
    Class to simulate behavior of sys.stdout or sys.stderr.
    Stores contents in internal buffer and optionally echos to the inner stream it is simulating.
    """

    def __init__(self, inner_stream: Union[TextIO, 'StdSim'], *, echo: bool=False, encoding: str='utf-8', errors: str='replace') -> None:
        """
        StdSim Initializer

        :param inner_stream: the wrapped stream. Should be a TextIO or StdSim instance.
        :param echo: if True, then all input will be echoed to inner_stream
        :param encoding: codec for encoding/decoding strings (defaults to utf-8)
        :param errors: how to handle encoding/decoding errors (defaults to replace)
        """
        self.inner_stream = inner_stream
        self.echo = echo
        self.encoding = encoding
        self.errors = errors
        self.pause_storage = False
        self.buffer = ByteBuf(self)

    def write(self, s: str) -> None:
        """
        Add str to internal bytes buffer and if echo is True, echo contents to inner stream

        :param s: String to write to the stream
        """
        if not isinstance(s, str):
            raise TypeError(f'write() argument must be str, not {type(s)}')
        if not self.pause_storage:
            self.buffer.byte_buf += s.encode(encoding=self.encoding, errors=self.errors)
        if self.echo:
            self.inner_stream.write(s)

    def getvalue(self) -> str:
        """Get the internal contents as a str"""
        return self.buffer.byte_buf.decode(encoding=self.encoding, errors=self.errors)

    def getbytes(self) -> bytes:
        """Get the internal contents as bytes"""
        return bytes(self.buffer.byte_buf)

    def read(self, size: Optional[int]=-1) -> str:
        """
        Read from the internal contents as a str and then clear them out

        :param size: Number of bytes to read from the stream
        """
        if size is None or size == -1:
            result = self.getvalue()
            self.clear()
        else:
            result = self.buffer.byte_buf[:size].decode(encoding=self.encoding, errors=self.errors)
            self.buffer.byte_buf = self.buffer.byte_buf[size:]
        return result

    def readbytes(self) -> bytes:
        """Read from the internal contents as bytes and then clear them out"""
        result = self.getbytes()
        self.clear()
        return result

    def clear(self) -> None:
        """Clear the internal contents"""
        self.buffer.byte_buf.clear()

    def isatty(self) -> bool:
        """StdSim only considered an interactive stream if `echo` is True and `inner_stream` is a tty."""
        if self.echo:
            return self.inner_stream.isatty()
        else:
            return False

    @property
    def line_buffering(self) -> bool:
        """
        Handle when the inner stream doesn't have a line_buffering attribute which is the case
        when running unit tests because pytest sets stdout to a pytest EncodedFile object.
        """
        try:
            return bool(self.inner_stream.line_buffering)
        except AttributeError:
            return False

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return getattr(self.inner_stream, item)