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
class Changes(_gpg_multivalued, _VersionAccessorMixin):
    """ Representation of a .changes (archive changes) file

    This class is a thin wrapper around the transparent GPG handling
    of :class:`_gpg_multivalued` and the parsing of :class:`Deb822`.
    """
    _multivalued_fields = {'files': ['md5sum', 'size', 'section', 'priority', 'name'], 'checksums-sha1': ['sha1', 'size', 'name'], 'checksums-sha256': ['sha256', 'size', 'name'], 'checksums-sha512': ['sha512', 'size', 'name']}

    def get_pool_path(self):
        """Return the path in the pool where the files would be installed"""
        s = self['files'][0]['section']
        try:
            section, _ = s.split('/')
        except ValueError:
            section = 'main'
        if self['source'].startswith('lib'):
            subdir = self['source'][:4]
        else:
            subdir = self['source'][0]
        return 'pool/%s/%s/%s' % (section, subdir, self['source'])