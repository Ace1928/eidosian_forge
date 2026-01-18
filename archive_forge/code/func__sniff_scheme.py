import collections
import io
import locale
import logging
import os
import os.path as P
import pathlib
import urllib.parse
import warnings
import smart_open.local_file as so_file
import smart_open.compression as so_compression
from smart_open import doctools
from smart_open import transport
from smart_open.compression import register_compressor  # noqa: F401
from smart_open.utils import check_kwargs as _check_kwargs  # noqa: F401
from smart_open.utils import inspect_kwargs as _inspect_kwargs  # noqa: F401
def _sniff_scheme(uri_as_string):
    """Returns the scheme of the URL only, as a string."""
    if os.name == 'nt' and '://' not in uri_as_string:
        uri_as_string = 'file://' + uri_as_string
    return urllib.parse.urlsplit(uri_as_string).scheme