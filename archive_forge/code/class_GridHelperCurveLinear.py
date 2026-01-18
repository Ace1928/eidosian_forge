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
class GridHelperCurveLinear(grid_helper_curvelinear.GridHelperCurveLinear):

    def __init__(self, aux_trans, extremes, grid_locator1=None, grid_locator2=None, tick_formatter1=None, tick_formatter2=None):
        super().__init__(aux_trans, extreme_finder=ExtremeFinderFixed(extremes), grid_locator1=grid_locator1, grid_locator2=grid_locator2, tick_formatter1=tick_formatter1, tick_formatter2=tick_formatter2)

    @_api.deprecated('3.8')
    def get_data_boundary(self, side):
        """
        Return v=0, nth=1.
        """
        lon1, lon2, lat1, lat2 = self.grid_finder.extreme_finder(*[None] * 5)
        return dict(left=(lon1, 0), right=(lon2, 0), bottom=(lat1, 1), top=(lat2, 1))[side]

    def new_fixed_axis(self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    def _update_grid(self, x1, y1, x2, y2):
        if self._grid_info is None:
            self._grid_info = dict()
        grid_info = self._grid_info
        grid_finder = self.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy, x1, y1, x2, y2)
        lon_min, lon_max = sorted(extremes[:2])
        lat_min, lat_max = sorted(extremes[2:])
        grid_info['extremes'] = (lon_min, lon_max, lat_min, lat_max)
        lon_levs, lon_n, lon_factor = grid_finder.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = grid_finder.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)
        grid_info['lon_info'] = (lon_levs, lon_n, lon_factor)
        grid_info['lat_info'] = (lat_levs, lat_n, lat_factor)
        grid_info['lon_labels'] = grid_finder.tick_formatter1('bottom', lon_factor, lon_levs)
        grid_info['lat_labels'] = grid_finder.tick_formatter2('bottom', lat_factor, lat_levs)
        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor
        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(lon_values[(lon_min < lon_values) & (lon_values < lon_max)], lat_values[(lat_min < lat_values) & (lat_values < lat_max)], lon_min, lon_max, lat_min, lat_max)
        grid_info['lon_lines'] = lon_lines
        grid_info['lat_lines'] = lat_lines
        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(extremes[:2], extremes[2:], *extremes)
        grid_info['lon_lines0'] = lon_lines
        grid_info['lat_lines0'] = lat_lines

    def get_gridlines(self, which='major', axis='both'):
        grid_lines = []
        if axis in ['both', 'x']:
            grid_lines.extend(self._grid_info['lon_lines'])
        if axis in ['both', 'y']:
            grid_lines.extend(self._grid_info['lat_lines'])
        return grid_lines