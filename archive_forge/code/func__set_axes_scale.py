import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def _set_axes_scale(self, value, **kwargs):
    """
        Set this Axis' scale.

        Parameters
        ----------
        value : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
            The axis scale type to apply.

        **kwargs
            Different keyword arguments are accepted, depending on the scale.
            See the respective class keyword arguments:

            - `matplotlib.scale.LinearScale`
            - `matplotlib.scale.LogScale`
            - `matplotlib.scale.SymmetricalLogScale`
            - `matplotlib.scale.LogitScale`
            - `matplotlib.scale.FuncScale`

        Notes
        -----
        By default, Matplotlib supports the above-mentioned scales.
        Additionally, custom scales may be registered using
        `matplotlib.scale.register_scale`. These scales can then also
        be used here.
        """
    name = self._get_axis_name()
    old_default_lims = self.get_major_locator().nonsingular(-np.inf, np.inf)
    for ax in self._get_shared_axes():
        ax._axis_map[name]._set_scale(value, **kwargs)
        ax._update_transScale()
        ax.stale = True
    new_default_lims = self.get_major_locator().nonsingular(-np.inf, np.inf)
    if old_default_lims != new_default_lims:
        self.axes.autoscale_view(**{f'scale{k}': k == name for k in self.axes._axis_names})