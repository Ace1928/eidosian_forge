import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
class MMDF(_mboxMMDF):
    """An MMDF mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize an MMDF mailbox."""
        self._message_factory = MMDFMessage
        _mboxMMDF.__init__(self, path, factory, create)

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        f.write(b'\x01\x01\x01\x01' + linesep)

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        f.write(linesep + b'\x01\x01\x01\x01' + linesep)

    def _generate_toc(self):
        """Generate key-to-(start, stop) table of contents."""
        starts, stops = ([], [])
        self._file.seek(0)
        next_pos = 0
        while True:
            line_pos = next_pos
            line = self._file.readline()
            next_pos = self._file.tell()
            if line.startswith(b'\x01\x01\x01\x01' + linesep):
                starts.append(next_pos)
                while True:
                    line_pos = next_pos
                    line = self._file.readline()
                    next_pos = self._file.tell()
                    if line == b'\x01\x01\x01\x01' + linesep:
                        stops.append(line_pos - len(linesep))
                        break
                    elif not line:
                        stops.append(line_pos)
                        break
            elif not line:
                break
        self._toc = dict(enumerate(zip(starts, stops)))
        self._next_key = len(self._toc)
        self._file.seek(0, 2)
        self._file_length = self._file.tell()