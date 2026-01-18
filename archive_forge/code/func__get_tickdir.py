import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _get_tickdir(self, position):
    """
        Get the direction of the tick.

        Parameters
        ----------
        position : str, optional : {'upper', 'lower', 'default'}
            The position of the axis.

        Returns
        -------
        tickdir : int
            Index which indicates which coordinate the tick line will
            align with.
        """
    _api.check_in_list(('upper', 'lower', 'default'), position=position)
    tickdirs_base = [v['tickdir'] for v in self._AXINFO.values()]
    elev_mod = np.mod(self.axes.elev + 180, 360) - 180
    azim_mod = np.mod(self.axes.azim, 360)
    if position == 'upper':
        if elev_mod >= 0:
            tickdirs_base = [2, 2, 0]
        else:
            tickdirs_base = [1, 0, 0]
        if 0 <= azim_mod < 180:
            tickdirs_base[2] = 1
    elif position == 'lower':
        if elev_mod >= 0:
            tickdirs_base = [1, 0, 1]
        else:
            tickdirs_base = [2, 2, 1]
        if 0 <= azim_mod < 180:
            tickdirs_base[2] = 0
    info_i = [v['i'] for v in self._AXINFO.values()]
    i = self._axinfo['i']
    vert_ax = self.axes._vertical_axis
    j = vert_ax - 2
    tickdir = np.roll(info_i, -j)[np.roll(tickdirs_base, j)][i]
    return tickdir