from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class MouseEnter(PointEvent):
    """ Announce a mouse enter event onto a Bokeh plot.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    .. note::
        The enter event is generated when the mouse leaves the entire Plot
        canvas, including any border padding and space for axes or legends.

    """
    event_name = 'mouseenter'