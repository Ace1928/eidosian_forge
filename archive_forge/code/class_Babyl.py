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
class Babyl(_singlefileMailbox):
    """An Rmail-style Babyl mailbox."""
    _special_labels = frozenset({'unseen', 'deleted', 'filed', 'answered', 'forwarded', 'edited', 'resent'})

    def __init__(self, path, factory=None, create=True):
        """Initialize a Babyl mailbox."""
        _singlefileMailbox.__init__(self, path, factory, create)
        self._labels = {}

    def add(self, message):
        """Add message and return assigned key."""
        key = _singlefileMailbox.add(self, message)
        if isinstance(message, BabylMessage):
            self._labels[key] = message.get_labels()
        return key

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        _singlefileMailbox.remove(self, key)
        if key in self._labels:
            del self._labels[key]

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        _singlefileMailbox.__setitem__(self, key, message)
        if isinstance(message, BabylMessage):
            self._labels[key] = message.get_labels()

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        self._file.readline()
        original_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == b'*** EOOH ***' + linesep or not line:
                break
            original_headers.write(line.replace(linesep, b'\n'))
        visible_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == linesep or not line:
                break
            visible_headers.write(line.replace(linesep, b'\n'))
        n = stop - self._file.tell()
        assert n >= 0
        body = self._file.read(n)
        body = body.replace(linesep, b'\n')
        msg = BabylMessage(original_headers.getvalue() + body)
        msg.set_visible(visible_headers.getvalue())
        if key in self._labels:
            msg.set_labels(self._labels[key])
        return msg

    def get_bytes(self, key):
        """Return a string representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        self._file.readline()
        original_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == b'*** EOOH ***' + linesep or not line:
                break
            original_headers.write(line.replace(linesep, b'\n'))
        while True:
            line = self._file.readline()
            if line == linesep or not line:
                break
        headers = original_headers.getvalue()
        n = stop - self._file.tell()
        assert n >= 0
        data = self._file.read(n)
        data = data.replace(linesep, b'\n')
        return headers + data

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        return io.BytesIO(self.get_bytes(key).replace(b'\n', linesep))

    def get_labels(self):
        """Return a list of user-defined labels in the mailbox."""
        self._lookup()
        labels = set()
        for label_list in self._labels.values():
            labels.update(label_list)
        labels.difference_update(self._special_labels)
        return list(labels)

    def _generate_toc(self):
        """Generate key-to-(start, stop) table of contents."""
        starts, stops = ([], [])
        self._file.seek(0)
        next_pos = 0
        label_lists = []
        while True:
            line_pos = next_pos
            line = self._file.readline()
            next_pos = self._file.tell()
            if line == b'\x1f\x0c' + linesep:
                if len(stops) < len(starts):
                    stops.append(line_pos - len(linesep))
                starts.append(next_pos)
                labels = [label.strip() for label in self._file.readline()[1:].split(b',') if label.strip()]
                label_lists.append(labels)
            elif line == b'\x1f' or line == b'\x1f' + linesep:
                if len(stops) < len(starts):
                    stops.append(line_pos - len(linesep))
            elif not line:
                stops.append(line_pos - len(linesep))
                break
        self._toc = dict(enumerate(zip(starts, stops)))
        self._labels = dict(enumerate(label_lists))
        self._next_key = len(self._toc)
        self._file.seek(0, 2)
        self._file_length = self._file.tell()

    def _pre_mailbox_hook(self, f):
        """Called before writing the mailbox to file f."""
        babyl = b'BABYL OPTIONS:' + linesep
        babyl += b'Version: 5' + linesep
        labels = self.get_labels()
        labels = (label.encode() for label in labels)
        babyl += b'Labels:' + b','.join(labels) + linesep
        babyl += b'\x1f'
        f.write(babyl)

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        f.write(b'\x0c' + linesep)

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        f.write(linesep + b'\x1f')

    def _install_message(self, message):
        """Write message contents and return (start, stop)."""
        start = self._file.tell()
        if isinstance(message, BabylMessage):
            special_labels = []
            labels = []
            for label in message.get_labels():
                if label in self._special_labels:
                    special_labels.append(label)
                else:
                    labels.append(label)
            self._file.write(b'1')
            for label in special_labels:
                self._file.write(b', ' + label.encode())
            self._file.write(b',,')
            for label in labels:
                self._file.write(b' ' + label.encode() + b',')
            self._file.write(linesep)
        else:
            self._file.write(b'1,,' + linesep)
        if isinstance(message, email.message.Message):
            orig_buffer = io.BytesIO()
            orig_generator = email.generator.BytesGenerator(orig_buffer, False, 0)
            orig_generator.flatten(message)
            orig_buffer.seek(0)
            while True:
                line = orig_buffer.readline()
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    break
            self._file.write(b'*** EOOH ***' + linesep)
            if isinstance(message, BabylMessage):
                vis_buffer = io.BytesIO()
                vis_generator = email.generator.BytesGenerator(vis_buffer, False, 0)
                vis_generator.flatten(message.get_visible())
                while True:
                    line = vis_buffer.readline()
                    self._file.write(line.replace(b'\n', linesep))
                    if line == b'\n' or not line:
                        break
            else:
                orig_buffer.seek(0)
                while True:
                    line = orig_buffer.readline()
                    self._file.write(line.replace(b'\n', linesep))
                    if line == b'\n' or not line:
                        break
            while True:
                buffer = orig_buffer.read(4096)
                if not buffer:
                    break
                self._file.write(buffer.replace(b'\n', linesep))
        elif isinstance(message, (bytes, str, io.StringIO)):
            if isinstance(message, io.StringIO):
                warnings.warn('Use of StringIO input is deprecated, use BytesIO instead', DeprecationWarning, 3)
                message = message.getvalue()
            if isinstance(message, str):
                message = self._string_to_bytes(message)
            body_start = message.find(b'\n\n') + 2
            if body_start - 2 != -1:
                self._file.write(message[:body_start].replace(b'\n', linesep))
                self._file.write(b'*** EOOH ***' + linesep)
                self._file.write(message[:body_start].replace(b'\n', linesep))
                self._file.write(message[body_start:].replace(b'\n', linesep))
            else:
                self._file.write(b'*** EOOH ***' + linesep + linesep)
                self._file.write(message.replace(b'\n', linesep))
        elif hasattr(message, 'readline'):
            if hasattr(message, 'buffer'):
                warnings.warn('Use of text mode files is deprecated, use a binary mode file instead', DeprecationWarning, 3)
                message = message.buffer
            original_pos = message.tell()
            first_pass = True
            while True:
                line = message.readline()
                if line.endswith(b'\r\n'):
                    line = line[:-2] + b'\n'
                elif line.endswith(b'\r'):
                    line = line[:-1] + b'\n'
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    if first_pass:
                        first_pass = False
                        self._file.write(b'*** EOOH ***' + linesep)
                        message.seek(original_pos)
                    else:
                        break
            while True:
                line = message.readline()
                if not line:
                    break
                if line.endswith(b'\r\n'):
                    line = line[:-2] + linesep
                elif line.endswith(b'\r'):
                    line = line[:-1] + linesep
                elif line.endswith(b'\n'):
                    line = line[:-1] + linesep
                self._file.write(line)
        else:
            raise TypeError('Invalid message type: %s' % type(message))
        stop = self._file.tell()
        return (start, stop)