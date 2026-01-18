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
def _encoding_wrapper(fileobj, mode, encoding=None, errors=None, newline=None):
    """Decode bytes into text, if necessary.

    If mode specifies binary access, does nothing, unless the encoding is
    specified.  A non-null encoding implies text mode.

    :arg fileobj: must quack like a filehandle object.
    :arg str mode: is the mode which was originally requested by the user.
    :arg str encoding: The text encoding to use.  If mode is binary, overrides mode.
    :arg str errors: The method to use when handling encoding/decoding errors.
    :returns: a file object
    """
    logger.debug('encoding_wrapper: %r', locals())
    if 'b' in mode and encoding is None:
        return fileobj
    if encoding is None:
        encoding = DEFAULT_ENCODING
    fileobj = io.TextIOWrapper(fileobj, encoding=encoding, errors=errors, newline=newline, write_through=True)
    return fileobj