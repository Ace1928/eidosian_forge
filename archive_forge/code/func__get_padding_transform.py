import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _get_padding_transform(self, padding_angle, loc, xylabel, padding_factor=1):
    """Get transform from angle and padding for non-inline labels"""
    if self.rotate_labels is False and loc != 'geo':
        padding_angle = {'top': 90.0, 'right': 0.0, 'bottom': -90.0, 'left': 180.0}[loc]
    if xylabel == 'x':
        padding = self.xpadding if self.xpadding is not None else matplotlib.rcParams['xtick.major.pad']
    else:
        padding = self.ypadding if self.ypadding is not None else matplotlib.rcParams['ytick.major.pad']
    dx = padding_factor * padding * np.cos(padding_angle * np.pi / 180)
    dy = padding_factor * padding * np.sin(padding_angle * np.pi / 180)
    return mtrans.offset_copy(self.axes.transData, fig=self.axes.figure, x=dx, y=dy, units='points')