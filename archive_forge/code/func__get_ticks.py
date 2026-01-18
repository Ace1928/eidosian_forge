import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def _get_ticks(self, axis, ticker):
    axis_props = {}
    if isinstance(ticker, (tuple, list)):
        if all((isinstance(t, tuple) for t in ticker)):
            ticks, labels = zip(*ticker)
            labels = [l if isinstance(l, str) else str(l) for l in labels]
            axis_props['tickvals'] = ticks
            axis_props['ticktext'] = labels
        else:
            axis_props['tickvals'] = ticker
        axis.update(axis_props)