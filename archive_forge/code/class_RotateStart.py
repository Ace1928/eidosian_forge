from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class RotateStart(PointEvent):
    """ Announce the start of a rotate event on a Bokeh plot.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    .. note::
        This event is only applicable for touch-enabled devices.

    """
    event_name = 'rotatestart'