from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def _line_ranges(statements: Iterable[TLineNo], lines: Iterable[TLineNo]) -> list[tuple[TLineNo, TLineNo]]:
    """Produce a list of ranges for `format_lines`."""
    statements = sorted(statements)
    lines = sorted(lines)
    pairs = []
    start = None
    lidx = 0
    for stmt in statements:
        if lidx >= len(lines):
            break
        if stmt == lines[lidx]:
            lidx += 1
            if not start:
                start = stmt
            end = stmt
        elif start:
            pairs.append((start, end))
            start = None
    if start:
        pairs.append((start, end))
    return pairs