import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
class ZoomLevel:
    """ Generates the ticks which appear in a specific zoom level """

    def __init__(self, tickSpecs, exampleText):
        """
        ============= ==========================================================
        tickSpecs     a list of one or more TickSpec objects with decreasing
                      coarseness
        ============= ==========================================================

        """
        self.tickSpecs = tickSpecs
        self.utcOffset = 0
        self.exampleText = exampleText

    def tickValues(self, minVal, maxVal, minSpc):
        allTicks = np.array([])
        valueSpecs = []
        utcMin = minVal - self.utcOffset
        utcMax = maxVal - self.utcOffset
        for spec in self.tickSpecs:
            ticks, skipFactor = spec.makeTicks(utcMin, utcMax, minSpc)
            ticks += self.utcOffset
            close = np.any(np.isclose(allTicks, ticks[:, np.newaxis], rtol=0, atol=minSpc * 0.01), axis=-1)
            ticks = ticks[~close]
            allTicks = np.concatenate([allTicks, ticks])
            valueSpecs.append((spec.spacing, ticks.tolist()))
            if skipFactor > 1:
                break
        return valueSpecs