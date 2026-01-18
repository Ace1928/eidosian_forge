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
def _get_binary_mode(mode_str):
    mode = list(mode_str)
    binmode = []
    if 't' in mode and 'b' in mode:
        raise ValueError("can't have text and binary mode at once")
    counts = [mode.count(x) for x in 'rwa']
    if sum(counts) > 1:
        raise ValueError('must have exactly one of create/read/write/append mode')

    def transfer(char):
        binmode.append(mode.pop(mode.index(char)))
    if 'a' in mode:
        transfer('a')
    elif 'w' in mode:
        transfer('w')
    elif 'r' in mode:
        transfer('r')
    else:
        raise ValueError('Must have exactly one of create/read/write/append mode and at most one plus')
    if 'b' in mode:
        transfer('b')
    elif 't' in mode:
        mode.pop(mode.index('t'))
        binmode.append('b')
    else:
        binmode.append('b')
    if '+' in mode:
        transfer('+')
    if mode:
        raise ValueError('invalid mode: %r' % mode_str)
    return ''.join(binmode)