import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class GpgInfo(_BaseGpgInfo):
    """A wrapper around gnupg parsable output obtained via --status-fd

    This class is really a dictionary containing parsed output from gnupg plus
    some methods to make sense of the data.
    Keys are keywords and values are arguments suitably split.
    See /usr/share/doc/gnupg/DETAILS.gz"""
    uidkeys = ('GOODSIG', 'EXPSIG', 'EXPKEYSIG', 'REVKEYSIG', 'BADSIG')

    def __init__(self, *args, **kwargs):
        super(GpgInfo, self).__init__(*args, **kwargs)
        self.out = None
        self.err = None

    def valid(self):
        """Is the signature valid?"""
        return 'GOODSIG' in self or 'VALIDSIG' in self

    def uid(self):
        """Return the primary ID of the signee key, None is not available"""

    @classmethod
    def from_output(cls, out, err=None):
        """ Create a GpgInfo object based on the gpg or gpgv output

        Create a new GpgInfo object from gpg(v) --status-fd output (out) and
        optionally collect stderr as well (err).

        Both out and err can be lines in newline-terminated sequence or
        regular strings.
        """
        n = cls()
        if isinstance(out, str):
            n.out = out.split('\n')
        else:
            n.out = out
        if isinstance(err, str):
            n.err = err.split('\n')
        else:
            n.err = err
        header = '[GNUPG:] '
        for line in n.out:
            if not line.startswith(header):
                continue
            line = line[len(header):]
            line = line.strip('\n')
            s = line.find(' ')
            key = line[:s]
            if key in cls.uidkeys:
                value = line[s + 1:].split(' ', 1)
            else:
                value = line[s + 1:].split(' ')
            if key in ('NEWSI', 'NEWSIG', 'KEY_CONSIDERED', 'PROGRESS'):
                continue
            n[key] = value
        return n

    @classmethod
    def from_sequence(cls, sequence, keyrings=None, executable=None):
        """Create a new GpgInfo object from the given sequence.

        :param sequence: sequence of lines of bytes or a single byte string

        :param keyrings: list of keyrings to use (default:
            ['/usr/share/keyrings/debian-keyring.gpg'])

        :param executable: list of args for subprocess.Popen, the first element
            being the gpgv executable (default: ['/usr/bin/gpgv'])
        """
        keyrings = keyrings or GPGV_DEFAULT_KEYRINGS
        executable = executable or [GPGV_EXECUTABLE]
        args = list(executable)
        args.extend(['--status-fd', '1'])
        for k in keyrings:
            args.extend(['--keyring', k])
        if '--keyring' not in args:
            raise IOError('cannot access any of the given keyrings')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=False) as p:
            if isinstance(sequence, bytes):
                inp = sequence
            else:
                inp = cls._get_full_bytes(sequence)
            out, err = p.communicate(inp)
        return cls.from_output(out.decode('utf-8'), err.decode('utf-8'))

    @staticmethod
    def _get_full_bytes(sequence):
        """Return a byte string from a sequence of lines of bytes.

        This method detects if the sequence's lines are newline-terminated, and
        constructs the byte string appropriately.
        """
        sequence_iter = iter(sequence)
        try:
            first_line = next(sequence_iter)
        except StopIteration:
            return b''
        join_str = b'\n'
        if first_line.endswith(b'\n'):
            join_str = b''
        return first_line + join_str + join_str.join(sequence_iter)

    @classmethod
    def from_file(cls, target, *args, **kwargs):
        """Create a new GpgInfo object from the given file.

        See GpgInfo.from_sequence.
        """
        with open(target, 'rb') as target_file:
            return cls.from_sequence(target_file, *args, **kwargs)