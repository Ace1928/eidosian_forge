from __future__ import absolute_import, unicode_literals
import io
import posixpath
import sys
from os import environ
from pybtex.exceptions import PybtexError
from pybtex.kpathsea import kpsewhich
def _decode_filename(filename, errors='strict'):
    """Decode byte-encoded filename."""
    encoding = sys.getfilesystemencoding() or get_default_encoding()
    return filename.decode(encoding, errors=errors)