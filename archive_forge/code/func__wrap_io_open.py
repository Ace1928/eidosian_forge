import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _wrap_io_open(file: t.Union[str, 'os.PathLike[str]', int], mode: str, encoding: t.Optional[str], errors: t.Optional[str]) -> t.IO[t.Any]:
    """Handles not passing ``encoding`` and ``errors`` in binary mode."""
    if 'b' in mode:
        return open(file, mode)
    return open(file, mode, encoding=encoding, errors=errors)