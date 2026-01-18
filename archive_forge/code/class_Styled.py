from typing import TYPE_CHECKING
from .measure import Measurement
from .segment import Segment
from .style import StyleType
class Styled:
    """Apply a style to a renderable.

    Args:
        renderable (RenderableType): Any renderable.
        style (StyleType): A style to apply across the entire renderable.
    """

    def __init__(self, renderable: 'RenderableType', style: 'StyleType') -> None:
        self.renderable = renderable
        self.style = style

    def __rich_console__(self, console: 'Console', options: 'ConsoleOptions') -> 'RenderResult':
        style = console.get_style(self.style)
        rendered_segments = console.render(self.renderable, options)
        segments = Segment.apply_style(rendered_segments, style)
        return segments

    def __rich_measure__(self, console: 'Console', options: 'ConsoleOptions') -> Measurement:
        return Measurement.get(console, options, self.renderable)