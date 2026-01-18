from .interpolatableHelpers import *
from .interpolatableTestContourOrder import test_contour_order
from .interpolatableTestStartingPoint import test_starting_point
from fontTools.pens.recordingPen import (
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.statisticsPen import StatisticsPen, StatisticsControlPen
from fontTools.pens.momentsPen import OpenContourError
from fontTools.varLib.models import piecewiseLinearMap, normalizeLocation
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.transform import Transform
from collections import defaultdict
from types import SimpleNamespace
from functools import wraps
from pprint import pformat
from math import sqrt, atan2, pi
import logging
import os
def _fill_in(self, ix):
    for item in self.ITEMS:
        if len(getattr(self, item)) == ix:
            getattr(self, item).append(None)