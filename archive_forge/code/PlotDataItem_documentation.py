import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem

        Returns the size in pixels that this item may draw beyond the values returned by dataBounds().
        This method is called by :class:`ViewBox` when auto-scaling.
        