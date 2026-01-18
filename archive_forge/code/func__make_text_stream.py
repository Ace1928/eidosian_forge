import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _make_text_stream(stream: t.BinaryIO, encoding: t.Optional[str], errors: t.Optional[str], force_readable: bool=False, force_writable: bool=False) -> t.TextIO:
    if encoding is None:
        encoding = get_best_encoding(stream)
    if errors is None:
        errors = 'replace'
    return _NonClosingTextIOWrapper(stream, encoding, errors, line_buffering=True, force_readable=force_readable, force_writable=force_writable)