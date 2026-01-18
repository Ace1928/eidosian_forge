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
def _reset_locator_formatter_scale(self):
    """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
    self._process_values()
    self._locator = None
    self._minorlocator = None
    self._formatter = None
    self._minorformatter = None
    if isinstance(self.mappable, contour.ContourSet) and isinstance(self.norm, colors.LogNorm):
        self._set_scale('log')
    elif self.boundaries is not None or isinstance(self.norm, colors.BoundaryNorm):
        if self.spacing == 'uniform':
            funcs = (self._forward_boundaries, self._inverse_boundaries)
            self._set_scale('function', functions=funcs)
        elif self.spacing == 'proportional':
            self._set_scale('linear')
    elif getattr(self.norm, '_scale', None):
        self._set_scale(self.norm._scale)
    elif type(self.norm) is colors.Normalize:
        self._set_scale('linear')
    else:
        funcs = (self.norm, self.norm.inverse)
        self._set_scale('function', functions=funcs)