from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class DoubleTap(PointEvent):
    """ Announce a double-tap or double-click event on a Bokeh plot.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    """
    event_name = 'doubletap'