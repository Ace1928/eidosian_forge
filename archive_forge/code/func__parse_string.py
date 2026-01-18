import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _parse_string(value: bytes) -> bytes:
    value = bytearray(value.strip())
    ret = bytearray()
    whitespace = bytearray()
    in_quotes = False
    i = 0
    while i < len(value):
        c = value[i]
        if c == ord(b'\\'):
            i += 1
            try:
                v = _ESCAPE_TABLE[value[i]]
            except IndexError as exc:
                raise ValueError('escape character in %r at %d before end of string' % (value, i)) from exc
            except KeyError as exc:
                raise ValueError('escape character followed by unknown character %s at %d in %r' % (value[i], i, value)) from exc
            if whitespace:
                ret.extend(whitespace)
                whitespace = bytearray()
            ret.append(v)
        elif c == ord(b'"'):
            in_quotes = not in_quotes
        elif c in _COMMENT_CHARS and (not in_quotes):
            break
        elif c in _WHITESPACE_CHARS:
            whitespace.append(c)
        else:
            if whitespace:
                ret.extend(whitespace)
                whitespace = bytearray()
            ret.append(c)
        i += 1
    if in_quotes:
        raise ValueError('missing end quote')
    return bytes(ret)