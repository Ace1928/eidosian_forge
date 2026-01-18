from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def _pprint_seq(seq: Sequence, _nest_lvl: int=0, max_seq_items: int | None=None, **kwds) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
    if isinstance(seq, set):
        fmt = '{{{body}}}'
    else:
        fmt = '[{body}]' if hasattr(seq, '__setitem__') else '({body})'
    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option('max_seq_items') or len(seq)
    s = iter(seq)
    r = [pprint_thing(next(s), _nest_lvl + 1, max_seq_items=max_seq_items, **kwds) for i in range(min(nitems, len(seq)))]
    body = ', '.join(r)
    if nitems < len(seq):
        body += ', ...'
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ','
    return fmt.format(body=body)