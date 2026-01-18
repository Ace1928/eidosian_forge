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
class PdiffIndex(_multivalued):
    """ Representation of a foo.diff/Index file from a Debian mirror

    This class is a thin wrapper around the transparent GPG handling
    of :class:`_gpg_multivalued` and the parsing of :class:`Deb822`.
    """
    _multivalued_fields = {'sha1-current': ['SHA1', 'size'], 'sha1-history': ['SHA1', 'size', 'date'], 'sha1-patches': ['SHA1', 'size', 'date'], 'sha1-download': ['SHA1', 'size', 'filename'], 'x-unmerged-sha1-history': ['SHA1', 'size', 'date'], 'x-unmerged-sha1-patches': ['SHA1', 'size', 'date'], 'x-unmerged-sha1-download': ['SHA1', 'size', 'filename'], 'sha256-current': ['SHA256', 'size'], 'sha256-history': ['SHA256', 'size', 'date'], 'sha256-patches': ['SHA256', 'size', 'date'], 'sha256-download': ['SHA256', 'size', 'filename'], 'x-unmerged-sha256-history': ['SHA256', 'size', 'date'], 'x-unmerged-sha256-patches': ['SHA256', 'size', 'date'], 'x-unmerged-sha256-download': ['SHA256', 'size', 'filename']}

    @property
    def _fixed_field_lengths(self):
        fixed_field_lengths = {}
        for key in self._multivalued_fields:
            if hasattr(self[key], 'keys'):
                continue
            length = self._get_size_field_length(key)
            fixed_field_lengths[key] = {'size': length}
        return fixed_field_lengths

    def _get_size_field_length(self, key):
        lengths = [len(str(item['size'])) for item in self[key]]
        return max(lengths)