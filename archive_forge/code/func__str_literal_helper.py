import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _str_literal_helper(self, string, *, quote_types=_ALL_QUOTES, escape_special_whitespace=False):
    """Helper for writing string literals, minimizing escapes.
        Returns the tuple (string literal to write, possible quote types).
        """

    def escape_char(c):
        if not escape_special_whitespace and c in '\n\t':
            return c
        if c == '\\' or not c.isprintable():
            return c.encode('unicode_escape').decode('ascii')
        return c
    escaped_string = ''.join(map(escape_char, string))
    possible_quotes = quote_types
    if '\n' in escaped_string:
        possible_quotes = [q for q in possible_quotes if q in _MULTI_QUOTES]
    possible_quotes = [q for q in possible_quotes if q not in escaped_string]
    if not possible_quotes:
        string = repr(string)
        quote = next((q for q in quote_types if string[0] in q), string[0])
        return (string[1:-1], [quote])
    if escaped_string:
        possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
        if possible_quotes[0][0] == escaped_string[-1]:
            assert len(possible_quotes[0]) == 3
            escaped_string = escaped_string[:-1] + '\\' + escaped_string[-1]
    return (escaped_string, possible_quotes)