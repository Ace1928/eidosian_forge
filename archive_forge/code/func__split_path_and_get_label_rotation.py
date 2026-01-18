from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=5):
    """
        Prepare for insertion of a label at index *idx* of *path*.

        Parameters
        ----------
        path : Path
            The path where the label will be inserted, in data space.
        idx : int
            The vertex index after which the label will be inserted.
        screen_pos : (float, float)
            The position where the label will be inserted, in screen space.
        lw : float
            The label width, in screen space.
        spacing : float
            Extra spacing around the label, in screen space.

        Returns
        -------
        path : Path
            The path, broken so that the label can be drawn over it.
        angle : float
            The rotation of the label.

        Notes
        -----
        Both tasks are done together to avoid calculating path lengths multiple times,
        which is relatively costly.

        The method used here involves computing the path length along the contour in
        pixel coordinates and then looking (label width / 2) away from central point to
        determine rotation and then to break contour if desired.  The extra spacing is
        taken into account when breaking the path, but not when computing the angle.
        """
    if hasattr(self, '_old_style_split_collections'):
        vis = False
        for coll in self._old_style_split_collections:
            vis |= coll.get_visible()
            coll.remove()
        self.set_visible(vis)
        del self._old_style_split_collections
    xys = path.vertices
    codes = path.codes
    pos = self.get_transform().inverted().transform(screen_pos)
    if not np.allclose(pos, xys[idx]):
        xys = np.insert(xys, idx, pos, axis=0)
        codes = np.insert(codes, idx, Path.LINETO)
    movetos = (codes == Path.MOVETO).nonzero()[0]
    start = movetos[movetos <= idx][-1]
    try:
        stop = movetos[movetos > idx][0]
    except IndexError:
        stop = len(codes)
    cc_xys = xys[start:stop]
    idx -= start
    is_closed_path = codes[stop - 1] == Path.CLOSEPOLY
    if is_closed_path:
        cc_xys = np.concatenate([cc_xys[idx:-1], cc_xys[:idx + 1]])
        idx = 0

    def interp_vec(x, xp, fp):
        return [np.interp(x, xp, col) for col in fp.T]
    screen_xys = self.get_transform().transform(cc_xys)
    path_cpls = np.insert(np.cumsum(np.hypot(*np.diff(screen_xys, axis=0).T)), 0, 0)
    path_cpls -= path_cpls[idx]
    target_cpls = np.array([-lw / 2, lw / 2])
    if is_closed_path:
        target_cpls[0] += path_cpls[-1] - path_cpls[0]
    (sx0, sx1), (sy0, sy1) = interp_vec(target_cpls, path_cpls, screen_xys)
    angle = np.rad2deg(np.arctan2(sy1 - sy0, sx1 - sx0))
    if self.rightside_up:
        angle = (angle + 90) % 180 - 90
    target_cpls += [-spacing, +spacing]
    i0, i1 = np.interp(target_cpls, path_cpls, range(len(path_cpls)), left=-1, right=-1)
    i0 = math.floor(i0)
    i1 = math.ceil(i1)
    (x0, x1), (y0, y1) = interp_vec(target_cpls, path_cpls, cc_xys)
    new_xy_blocks = []
    new_code_blocks = []
    if is_closed_path:
        if i0 != -1 and i1 != -1:
            points = cc_xys[i1:i0 + 1]
            new_xy_blocks.extend([[(x1, y1)], points, [(x0, y0)]])
            nlines = len(points) + 1
            new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * nlines])
    else:
        if i0 != -1:
            new_xy_blocks.extend([cc_xys[:i0 + 1], [(x0, y0)]])
            new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (i0 + 1)])
        if i1 != -1:
            new_xy_blocks.extend([[(x1, y1)], cc_xys[i1:]])
            new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (len(cc_xys) - i1)])
    xys = np.concatenate([xys[:start], *new_xy_blocks, xys[stop:]])
    codes = np.concatenate([codes[:start], *new_code_blocks, codes[stop:]])
    return (angle, Path(xys, codes))