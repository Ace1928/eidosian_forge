from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
class SegmentLines:

    def __init__(self, lines: Iterable[List[Segment]], new_lines: bool=False) -> None:
        """A simple renderable containing a number of lines of segments. May be used as an intermediate
        in rendering process.

        Args:
            lines (Iterable[List[Segment]]): Lists of segments forming lines.
            new_lines (bool, optional): Insert new lines after each line. Defaults to False.
        """
        self.lines = list(lines)
        self.new_lines = new_lines

    def __rich_console__(self, console: 'Console', options: 'ConsoleOptions') -> 'RenderResult':
        if self.new_lines:
            new_line = Segment.line()
            for line in self.lines:
                yield from line
                yield new_line
        else:
            for line in self.lines:
                yield from line