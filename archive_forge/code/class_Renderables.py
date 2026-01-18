from itertools import zip_longest
from typing import (
from .cells import cell_len
from .measure import Measurement
class Renderables:
    """A list subclass which renders its contents to the console."""

    def __init__(self, renderables: Optional[Iterable['RenderableType']]=None) -> None:
        self._renderables: List['RenderableType'] = list(renderables) if renderables is not None else []

    def __rich_console__(self, console: 'Console', options: 'ConsoleOptions') -> 'RenderResult':
        """Console render method to insert line-breaks."""
        yield from self._renderables

    def __rich_measure__(self, console: 'Console', options: 'ConsoleOptions') -> 'Measurement':
        dimensions = [Measurement.get(console, options, renderable) for renderable in self._renderables]
        if not dimensions:
            return Measurement(1, 1)
        _min = max((dimension.minimum for dimension in dimensions))
        _max = max((dimension.maximum for dimension in dimensions))
        return Measurement(_min, _max)

    def append(self, renderable: 'RenderableType') -> None:
        self._renderables.append(renderable)

    def __iter__(self) -> Iterable['RenderableType']:
        return iter(self._renderables)