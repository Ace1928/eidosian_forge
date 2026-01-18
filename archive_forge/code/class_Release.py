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
class Release(_multivalued):
    """Represents a Release file

    Set the size_field_behavior attribute to "dak" to make the size field
    length only as long as the longest actual value.  The default,
    "apt-ftparchive" makes the field 16 characters long regardless.

    This class is a thin wrapper around the parsing of :class:`Deb822`.
    """
    _multivalued_fields = {'md5sum': ['md5sum', 'size', 'name'], 'sha1': ['sha1', 'size', 'name'], 'sha256': ['sha256', 'size', 'name'], 'sha512': ['sha512', 'size', 'name']}
    __size_field_behavior = 'apt-ftparchive'

    def set_size_field_behavior(self, value):
        if value not in ['apt-ftparchive', 'dak']:
            raise ValueError("size_field_behavior must be either 'apt-ftparchive' or 'dak'")
        self.__size_field_behavior = value
    size_field_behavior = property(lambda self: self.__size_field_behavior, set_size_field_behavior)

    @property
    def _fixed_field_lengths(self):
        fixed_field_lengths = {}
        for key in self._multivalued_fields:
            length = self._get_size_field_length(key)
            fixed_field_lengths[key] = {'size': length}
        return fixed_field_lengths

    def _get_size_field_length(self, key):
        if self.size_field_behavior == 'apt-ftparchive':
            return 16
        if self.size_field_behavior == 'dak':
            lengths = [len(str(item['size'])) for item in self[key]]
            return max(lengths)
        raise ValueError('Illegal value for size_field_behavior')