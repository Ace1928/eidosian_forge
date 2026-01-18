from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class MouseMove(PointEvent):
    """ Announce a mouse movement event over a Bokeh plot.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    .. note::
        This event can fire at a very high rate, potentially increasing network
        traffic or CPU load.

    """
    event_name = 'mousemove'