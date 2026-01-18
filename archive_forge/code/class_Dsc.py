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
class Dsc(_gpg_multivalued, _VersionAccessorMixin):
    """ Representation of a .dsc (Debian Source Control) file

    This class is a thin wrapper around the transparent GPG handling
    of :class:`_gpg_multivalued` and the parsing of :class:`Deb822`.
    """
    _multivalued_fields = {'files': ['md5sum', 'size', 'name'], 'checksums-sha1': ['sha1', 'size', 'name'], 'checksums-sha256': ['sha256', 'size', 'name'], 'checksums-sha512': ['sha512', 'size', 'name']}