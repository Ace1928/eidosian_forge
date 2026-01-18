import math
import weakref
import numpy as np
from .. import colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from .LinearRegionItem import LinearRegionItem
from .PlotItem import PlotItem
def _regionChanging(self):
    """ internal: recalculate levels based on new position of adjusters """
    if not self.region_changed_enable:
        return
    bot, top = self.region.getRegion()
    bot = (bot - 63) / 64
    top = (top - 191) / 64
    bot = math.copysign(bot ** 2, bot)
    top = math.copysign(top ** 2, top)
    span_prv = self.hi_prv - self.lo_prv
    hi_new = self.hi_prv + (span_prv + 2 * self.rounding) * top
    lo_new = self.lo_prv + (span_prv + 2 * self.rounding) * bot
    if self.hi_lim is not None:
        if hi_new > self.hi_lim:
            hi_new = self.hi_lim
            if top != 0 and bot != 0:
                lo_new = hi_new - span_prv
    if self.lo_lim is not None:
        if lo_new < self.lo_lim:
            lo_new = self.lo_lim
            if top != 0 and bot != 0:
                hi_new = lo_new + span_prv
    if hi_new - lo_new < self.rounding:
        if bot == 0:
            hi_new = lo_new + self.rounding
        elif top == 0:
            lo_new = hi_new - self.rounding
        else:
            mid = (hi_new + lo_new) / 2
            hi_new = mid + self.rounding / 2
            lo_new = mid - self.rounding / 2
    lo_new = self.rounding * round(lo_new / self.rounding)
    hi_new = self.rounding * round(hi_new / self.rounding)
    self.values = (lo_new, hi_new)
    self._update_items()
    self.sigLevelsChanged.emit(self)