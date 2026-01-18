from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
class FStringNode:

    def __init__(self, quote):
        self.quote = quote
        self.parentheses_count = 0
        self.previous_lines = ''
        self.last_string_start_pos = None
        self.format_spec_count = 0

    def open_parentheses(self, character):
        self.parentheses_count += 1

    def close_parentheses(self, character):
        self.parentheses_count -= 1
        if self.parentheses_count == 0:
            self.format_spec_count = 0

    def allow_multiline(self):
        return len(self.quote) == 3

    def is_in_expr(self):
        return self.parentheses_count > self.format_spec_count

    def is_in_format_spec(self):
        return not self.is_in_expr() and self.format_spec_count