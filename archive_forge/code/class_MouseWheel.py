from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class MouseWheel(PointEvent):
    """ Announce a mouse wheel event on a Bokeh plot.

    Attributes:
        delta (float) : the (signed) scroll speed
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space


    .. note::
        By default, Bokeh plots do not prevent default scroll events unless a
        ``WheelZoomTool`` or ``WheelPanTool`` is active. This may change in
        future releases.

    """
    event_name = 'wheel'

    def __init__(self, model: Plot | None, *, delta: float | None=None, sx: float | None=None, sy: float | None=None, x: float | None=None, y: float | None=None, modifiers: KeyModifiers | None=None):
        self.delta = delta
        super().__init__(model, sx=sx, sy=sy, x=x, y=y, modifiers=modifiers)