import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _init3d(self):
    self.line = mlines.Line2D(xdata=(0, 0), ydata=(0, 0), linewidth=self._axinfo['axisline']['linewidth'], color=self._axinfo['axisline']['color'], antialiased=True)
    self.pane = mpatches.Polygon([[0, 0], [0, 1]], closed=False)
    self.set_pane_color(self._axinfo['color'])
    self.axes._set_artist_props(self.line)
    self.axes._set_artist_props(self.pane)
    self.gridlines = art3d.Line3DCollection([])
    self.axes._set_artist_props(self.gridlines)
    self.axes._set_artist_props(self.label)
    self.axes._set_artist_props(self.offsetText)
    self.label._transform = self.axes.transData
    self.offsetText._transform = self.axes.transData