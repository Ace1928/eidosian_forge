import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _byte_offset_to_character_offset(str, offset):
    as_utf8 = str.encode('utf-8')
    return len(as_utf8[:offset].decode('utf-8', errors='replace'))