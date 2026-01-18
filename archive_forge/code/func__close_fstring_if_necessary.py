from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def _close_fstring_if_necessary(fstring_stack, string, line_nr, column, additional_prefix):
    for fstring_stack_index, node in enumerate(fstring_stack):
        lstripped_string = string.lstrip()
        len_lstrip = len(string) - len(lstripped_string)
        if lstripped_string.startswith(node.quote):
            token = PythonToken(FSTRING_END, node.quote, (line_nr, column + len_lstrip), prefix=additional_prefix + string[:len_lstrip])
            additional_prefix = ''
            assert not node.previous_lines
            del fstring_stack[fstring_stack_index:]
            return (token, '', len(node.quote) + len_lstrip)
    return (None, additional_prefix, 0)