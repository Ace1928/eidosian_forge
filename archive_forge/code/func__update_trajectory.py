import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _update_trajectory(self, xm, ym, broken_streamlines=True):
    """
        Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
    if self._current_xy != (xm, ym):
        if self[ym, xm] == 0:
            self._traj.append((ym, xm))
            self._mask[ym, xm] = 1
            self._current_xy = (xm, ym)
        elif broken_streamlines:
            raise InvalidIndexError
        else:
            pass