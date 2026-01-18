from abc import ABC, abstractmethod
from itertools import islice
from operator import itemgetter
from threading import RLock
from typing import (
from ._ratio import ratio_resolve
from .align import Align
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .highlighter import ReprHighlighter
from .panel import Panel
from .pretty import Pretty
from .region import Region
from .repr import Result, rich_repr
from .segment import Segment
from .style import StyleType
class ColumnSplitter(Splitter):
    """Split a layout region in to columns."""
    name = 'column'

    def get_tree_icon(self) -> str:
        return '[layout.tree.column]⬍'

    def divide(self, children: Sequence['Layout'], region: Region) -> Iterable[Tuple['Layout', Region]]:
        x, y, width, height = region
        render_heights = ratio_resolve(height, children)
        offset = 0
        _Region = Region
        for child, child_height in zip(children, render_heights):
            yield (child, _Region(x, y + offset, width, child_height))
            offset += child_height