from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def _split_illegal_unicode_name(token, start_pos, prefix):

    def create_token():
        return PythonToken(ERRORTOKEN if is_illegal else NAME, found, pos, prefix)
    found = ''
    is_illegal = False
    pos = start_pos
    for i, char in enumerate(token):
        if is_illegal:
            if char.isidentifier():
                yield create_token()
                found = char
                is_illegal = False
                prefix = ''
                pos = (start_pos[0], start_pos[1] + i)
            else:
                found += char
        else:
            new_found = found + char
            if new_found.isidentifier():
                found = new_found
            else:
                if found:
                    yield create_token()
                    prefix = ''
                    pos = (start_pos[0], start_pos[1] + i)
                found = char
                is_illegal = True
    if found:
        yield create_token()