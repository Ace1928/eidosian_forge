import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
def _on_mouse(self, event):
    """Handle mpl mouse move and button press events"""
    if event.button != 1:
        return
    ii = self._in_axis(event)
    if ii is None:
        return
    if ii == 3:
        self._set_volume_index(event.xdata)
    else:
        xax, yax = [[1, 2], [0, 2], [0, 1]][ii]
        x, y = (event.xdata, event.ydata)
        x = self._sizes[xax] - x if self._flips[xax] else x
        y = self._sizes[yax] - y if self._flips[yax] else y
        idxs = [None, None, None, 1.0]
        idxs[xax] = x
        idxs[yax] = y
        idxs[ii] = self._data_idx[ii]
        self._set_position(*np.dot(self._affine, idxs)[:3])
    self._draw()