import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _add_solids(self, X, Y, C):
    """Draw the colors; optionally add separators."""
    if self.solids is not None:
        self.solids.remove()
    for solid in self.solids_patches:
        solid.remove()
    mappable = getattr(self, 'mappable', None)
    if isinstance(mappable, contour.ContourSet) and any((hatch is not None for hatch in mappable.hatches)):
        self._add_solids_patches(X, Y, C, mappable)
    else:
        self.solids = self.ax.pcolormesh(X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha, edgecolors='none', shading='flat')
        if not self.drawedges:
            if len(self._y) >= self.n_rasterize:
                self.solids.set_rasterized(True)
    self._update_dividers()