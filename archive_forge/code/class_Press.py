from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class Press(PointEvent):
    """ Announce a press event on a Bokeh plot.

    Attributes:
        sx (float) : x-coordinate of the event in *screen* space
        sy (float) : y-coordinate of the event in *screen* space
        x (float) : x-coordinate of the event in *data* space
        y (float) : y-coordinate of the event in *data* space

    """
    event_name = 'press'