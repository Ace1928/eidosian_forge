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
class _gpg_multivalued(_multivalued):
    """A _multivalued class that can support gpg signed objects

    This class's feature is that it stores the raw text before parsing so that
    gpg can verify the signature.  Use it just like you would use the
    _multivalued class.

    This class only stores raw text if it is given a raw string, or if it
    detects a gpg signature when given a file or sequence of lines (see
    Deb822.split_gpg_and_payload for details).
    """

    def __init__(self, *args, **kwargs):
        self.raw_text = None
        try:
            sequence = args[0]
        except IndexError:
            sequence = kwargs.get('sequence', None)
        strict = kwargs.get('strict', None)
        if sequence is not None:
            encoding = getattr(sequence, 'encoding', None) or kwargs.get('encoding', 'utf-8') or 'utf-8'
            if isinstance(sequence, bytes):
                self.raw_text = sequence
            elif isinstance(sequence, str):
                self.raw_text = sequence.encode(encoding)
            elif hasattr(sequence, 'items'):
                pass
            else:
                try:
                    gpg_pre_lines, lines, gpg_post_lines = self.split_gpg_and_payload((self._bytes(s, encoding) for s in sequence), strict)
                except EOFError:
                    gpg_pre_lines = lines = gpg_post_lines = []
                if gpg_pre_lines and gpg_post_lines:
                    raw_text = io.BytesIO()
                    raw_text.write(b'\n'.join(gpg_pre_lines))
                    raw_text.write(b'\n\n')
                    raw_text.write(b'\n'.join(lines))
                    raw_text.write(b'\n\n')
                    raw_text.write(b'\n'.join(gpg_post_lines))
                    self.raw_text = raw_text.getvalue()
                try:
                    argsl = list(args)
                    argsl[0] = lines
                    args = tuple(argsl)
                except IndexError:
                    kwargs['sequence'] = lines
        _multivalued.__init__(self, *args, **kwargs)

    @staticmethod
    def _bytes(s, encoding):
        """Converts s to bytes if necessary, using encoding.

        If s is already bytes type, returns it directly.
        """
        if isinstance(s, bytes):
            return s
        if isinstance(s, str):
            return s.encode(encoding)
        raise TypeError('bytes or unicode/string required, not %s' % type(s))