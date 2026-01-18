from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class LODEnd(PlotEvent):
    """ Announce the end of "interactive level-of-detail" mode on a plot.

    During interactive actions such as panning or zooming, Bokeh can
    optionally, temporarily draw a reduced set of the data, in order to
    maintain high interactive rates. This is referred to as interactive
    Level-of-Detail (LOD) mode. This event fires whenever a LOD mode
    has just ended.

    """
    event_name = 'lodend'