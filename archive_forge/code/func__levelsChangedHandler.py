import math
import weakref
import numpy as np
from .. import colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from .LinearRegionItem import LinearRegionItem
from .PlotItem import PlotItem
def _levelsChangedHandler(self, levels):
    """ internal: called when child item for some reason decides to update its levels without using ColorBarItem.
                      Will update colormap for the bar based on child items new levels """
    if levels != self.values:
        self.setLevels(levels, update_items=False)