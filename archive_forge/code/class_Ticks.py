from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
class Ticks(AttributeCopier, Line2D):
    """
    Ticks are derived from `.Line2D`, and note that ticks themselves
    are markers. Thus, you should use set_mec, set_mew, etc.

    To change the tick size (length), you need to use
    `set_ticksize`. To change the direction of the ticks (ticks are
    in opposite direction of ticklabels by default), use
    ``set_tick_out(False)``
    """

    def __init__(self, ticksize, tick_out=False, *, axis=None, **kwargs):
        self._ticksize = ticksize
        self.locs_angles_labels = []
        self.set_tick_out(tick_out)
        self._axis = axis
        if self._axis is not None:
            if 'color' not in kwargs:
                kwargs['color'] = 'auto'
            if 'mew' not in kwargs and 'markeredgewidth' not in kwargs:
                kwargs['markeredgewidth'] = 'auto'
        Line2D.__init__(self, [0.0], [0.0], **kwargs)
        self.set_snap(True)

    def get_ref_artist(self):
        return self._axis.majorTicks[0].tick1line

    def set_color(self, color):
        if not cbook._str_equal(color, 'auto'):
            mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def get_color(self):
        return self.get_attribute_from_ref_artist('color')

    def get_markeredgecolor(self):
        return self.get_attribute_from_ref_artist('markeredgecolor')

    def get_markeredgewidth(self):
        return self.get_attribute_from_ref_artist('markeredgewidth')

    def set_tick_out(self, b):
        """Set whether ticks are drawn inside or outside the axes."""
        self._tick_out = b

    def get_tick_out(self):
        """Return whether ticks are drawn inside or outside the axes."""
        return self._tick_out

    def set_ticksize(self, ticksize):
        """Set length of the ticks in points."""
        self._ticksize = ticksize

    def get_ticksize(self):
        """Return length of the ticks in points."""
        return self._ticksize

    def set_locs_angles(self, locs_angles):
        self.locs_angles = locs_angles
    _tickvert_path = Path([[0.0, 0.0], [1.0, 0.0]])

    def draw(self, renderer):
        if not self.get_visible():
            return
        gc = renderer.new_gc()
        gc.set_foreground(self.get_markeredgecolor())
        gc.set_linewidth(self.get_markeredgewidth())
        gc.set_alpha(self._alpha)
        path_trans = self.get_transform()
        marker_transform = Affine2D().scale(renderer.points_to_pixels(self._ticksize))
        if self.get_tick_out():
            marker_transform.rotate_deg(180)
        for loc, angle in self.locs_angles:
            locs = path_trans.transform_non_affine(np.array([loc]))
            if self.axes and (not self.axes.viewLim.contains(*locs[0])):
                continue
            renderer.draw_markers(gc, self._tickvert_path, marker_transform + Affine2D().rotate_deg(angle), Path(locs), path_trans.get_affine())
        gc.restore()