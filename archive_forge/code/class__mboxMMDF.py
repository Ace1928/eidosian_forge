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
class _mboxMMDF(_singlefileMailbox):
    """An mbox or MMDF mailbox."""
    _mangle_from_ = True

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        from_line = self._file.readline().replace(linesep, b'').decode('ascii')
        string = self._file.read(stop - self._file.tell())
        msg = self._message_factory(string.replace(linesep, b'\n'))
        msg.set_unixfrom(from_line)
        msg.set_from(from_line[5:])
        return msg

    def get_string(self, key, from_=False):
        """Return a string representation or raise a KeyError."""
        return email.message_from_bytes(self.get_bytes(key, from_)).as_string(unixfrom=from_)

    def get_bytes(self, key, from_=False):
        """Return a string representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        if not from_:
            self._file.readline()
        string = self._file.read(stop - self._file.tell())
        return string.replace(linesep, b'\n')

    def get_file(self, key, from_=False):
        """Return a file-like representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        if not from_:
            self._file.readline()
        return _PartialFile(self._file, self._file.tell(), stop)

    def _install_message(self, message):
        """Format a message and blindly write to self._file."""
        from_line = None
        if isinstance(message, str):
            message = self._string_to_bytes(message)
        if isinstance(message, bytes) and message.startswith(b'From '):
            newline = message.find(b'\n')
            if newline != -1:
                from_line = message[:newline]
                message = message[newline + 1:]
            else:
                from_line = message
                message = b''
        elif isinstance(message, _mboxMMDFMessage):
            author = message.get_from().encode('ascii')
            from_line = b'From ' + author
        elif isinstance(message, email.message.Message):
            from_line = message.get_unixfrom()
            if from_line is not None:
                from_line = from_line.encode('ascii')
        if from_line is None:
            from_line = b'From MAILER-DAEMON ' + time.asctime(time.gmtime()).encode()
        start = self._file.tell()
        self._file.write(from_line + linesep)
        self._dump_message(message, self._file, self._mangle_from_)
        stop = self._file.tell()
        return (start, stop)