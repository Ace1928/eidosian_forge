import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple
class FloatingAxesBase:

    def __init__(self, *args, grid_helper, **kwargs):
        _api.check_isinstance(GridHelperCurveLinear, grid_helper=grid_helper)
        super().__init__(*args, grid_helper=grid_helper, **kwargs)
        self.set_aspect(1.0)

    def _gen_axes_patch(self):
        x0, x1, y0, y1 = self.get_grid_helper().grid_finder.extreme_finder(*[None] * 5)
        patch = mpatches.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        patch.get_path()._interpolation_steps = 100
        return patch

    def clear(self):
        super().clear()
        self.patch.set_transform(self.get_grid_helper().grid_finder.get_transform() + self.transData)
        orig_patch = super()._gen_axes_patch()
        orig_patch.set_figure(self.figure)
        orig_patch.set_transform(self.transAxes)
        self.patch.set_clip_path(orig_patch)
        self.gridlines.set_clip_path(orig_patch)
        self.adjust_axes_lim()

    def adjust_axes_lim(self):
        bbox = self.patch.get_path().get_extents(self.patch.get_transform() - self.transData)
        bbox = bbox.expanded(1.02, 1.02)
        self.set_xlim(bbox.xmin, bbox.xmax)
        self.set_ylim(bbox.ymin, bbox.ymax)