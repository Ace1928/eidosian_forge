import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def _translate_segment(segment: bytes) -> bytes:
    if segment == b'*':
        return b'[^/]+'
    res = b''
    i, n = (0, len(segment))
    while i < n:
        c = segment[i:i + 1]
        i = i + 1
        if c == b'*':
            res += b'[^/]*'
        elif c == b'?':
            res += b'[^/]'
        elif c == b'\\':
            res += re.escape(segment[i:i + 1])
            i += 1
        elif c == b'[':
            j = i
            if j < n and segment[j:j + 1] == b'!':
                j = j + 1
            if j < n and segment[j:j + 1] == b']':
                j = j + 1
            while j < n and segment[j:j + 1] != b']':
                j = j + 1
            if j >= n:
                res += b'\\['
            else:
                stuff = segment[i:j].replace(b'\\', b'\\\\')
                i = j + 1
                if stuff.startswith(b'!'):
                    stuff = b'^' + stuff[1:]
                elif stuff.startswith(b'^'):
                    stuff = b'\\' + stuff
                res += b'[' + stuff + b']'
        else:
            res += re.escape(c)
    return res