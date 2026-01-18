import codecs
import io
import locale
import os
import sys
import unicodedata
from io import StringIO, BytesIO
def _get_exception_encoding():
    """Return the encoding we expect messages from the OS to be encoded in"""
    if os.name == 'nt':
        return 'mbcs'
    return locale.getlocale(locale.LC_MESSAGES)[1] or 'ascii'