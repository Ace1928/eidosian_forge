import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _get_code_position(code, instruction_index):
    if instruction_index < 0:
        return (None, None, None, None)
    positions_gen = code.co_positions()
    return next(itertools.islice(positions_gen, instruction_index // 2, None))