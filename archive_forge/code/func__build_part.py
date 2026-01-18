from base64 import standard_b64encode
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import zipfile
import tempfile
import shutil
import itertools
import functools
import http.client
import urllib.parse
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from .upload import upload
@staticmethod
def _build_part(item, sep_boundary):
    key, values = item
    title = '\nContent-Disposition: form-data; name="%s"' % key
    if not isinstance(values, list):
        values = [values]
    for value in values:
        if isinstance(value, tuple):
            title += '; filename="%s"' % value[0]
            value = value[1]
        else:
            value = _encode(value)
        yield sep_boundary
        yield _encode(title)
        yield b'\n\n'
        yield value
        if value and value[-1:] == b'\r':
            yield b'\n'