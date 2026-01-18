import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _display_width(line, offset):
    """Calculate the extra amount of width space the given source
    code segment might take if it were to be displayed on a fixed
    width output device. Supports wide unicode characters and emojis."""
    if line.isascii():
        return offset
    import unicodedata
    return sum((2 if unicodedata.east_asian_width(char) in _WIDE_CHAR_SPECIFIERS else 1 for char in line[:offset]))