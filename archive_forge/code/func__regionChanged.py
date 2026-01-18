import math
import weakref
import numpy as np
from .. import colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from .LinearRegionItem import LinearRegionItem
from .PlotItem import PlotItem
def _regionChanged(self):
    """ internal: snap adjusters back to default positions on release """
    self.lo_prv, self.hi_prv = self.values
    self.region_changed_enable = False
    self.region.setRegion((63, 191))
    self.region_changed_enable = True
    self.sigLevelsChangeFinished.emit(self)