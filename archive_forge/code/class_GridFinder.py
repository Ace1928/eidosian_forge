import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class GridFinder:
    """
    Internal helper for `~.grid_helper_curvelinear.GridHelperCurveLinear`, with
    the same constructor parameters; should not be directly instantiated.
    """

    def __init__(self, transform, extreme_finder=None, grid_locator1=None, grid_locator2=None, tick_formatter1=None, tick_formatter2=None):
        if extreme_finder is None:
            extreme_finder = ExtremeFinderSimple(20, 20)
        if grid_locator1 is None:
            grid_locator1 = MaxNLocator()
        if grid_locator2 is None:
            grid_locator2 = MaxNLocator()
        if tick_formatter1 is None:
            tick_formatter1 = FormatterPrettyPrint()
        if tick_formatter2 is None:
            tick_formatter2 = FormatterPrettyPrint()
        self.extreme_finder = extreme_finder
        self.grid_locator1 = grid_locator1
        self.grid_locator2 = grid_locator2
        self.tick_formatter1 = tick_formatter1
        self.tick_formatter2 = tick_formatter2
        self.set_transform(transform)

    def get_grid_info(self, x1, y1, x2, y2):
        """
        lon_values, lat_values : list of grid values. if integer is given,
                           rough number of grids in each direction.
        """
        extremes = self.extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)
        lon_min, lon_max, lat_min, lat_max = extremes
        lon_levs, lon_n, lon_factor = self.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = self.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)
        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor
        lon_lines, lat_lines = self._get_raw_grid_lines(lon_values, lat_values, lon_min, lon_max, lat_min, lat_max)
        ddx = (x2 - x1) * 1e-10
        ddy = (y2 - y1) * 1e-10
        bb = Bbox.from_extents(x1 - ddx, y1 - ddy, x2 + ddx, y2 + ddy)
        grid_info = {'extremes': extremes, 'lon_lines': lon_lines, 'lat_lines': lat_lines, 'lon': self._clip_grid_lines_and_find_ticks(lon_lines, lon_values, lon_levs, bb), 'lat': self._clip_grid_lines_and_find_ticks(lat_lines, lat_values, lat_levs, bb)}
        tck_labels = grid_info['lon']['tick_labels'] = {}
        for direction in ['left', 'bottom', 'right', 'top']:
            levs = grid_info['lon']['tick_levels'][direction]
            tck_labels[direction] = self.tick_formatter1(direction, lon_factor, levs)
        tck_labels = grid_info['lat']['tick_labels'] = {}
        for direction in ['left', 'bottom', 'right', 'top']:
            levs = grid_info['lat']['tick_levels'][direction]
            tck_labels[direction] = self.tick_formatter2(direction, lat_factor, levs)
        return grid_info

    def _get_raw_grid_lines(self, lon_values, lat_values, lon_min, lon_max, lat_min, lat_max):
        lons_i = np.linspace(lon_min, lon_max, 100)
        lats_i = np.linspace(lat_min, lat_max, 100)
        lon_lines = [self.transform_xy(np.full_like(lats_i, lon), lats_i) for lon in lon_values]
        lat_lines = [self.transform_xy(lons_i, np.full_like(lons_i, lat)) for lat in lat_values]
        return (lon_lines, lat_lines)

    def _clip_grid_lines_and_find_ticks(self, lines, values, levs, bb):
        gi = {'values': [], 'levels': [], 'tick_levels': dict(left=[], bottom=[], right=[], top=[]), 'tick_locs': dict(left=[], bottom=[], right=[], top=[]), 'lines': []}
        tck_levels = gi['tick_levels']
        tck_locs = gi['tick_locs']
        for (lx, ly), v, lev in zip(lines, values, levs):
            tcks = _find_line_box_crossings(np.column_stack([lx, ly]), bb)
            gi['levels'].append(v)
            gi['lines'].append([(lx, ly)])
            for tck, direction in zip(tcks, ['left', 'right', 'bottom', 'top']):
                for t in tck:
                    tck_levels[direction].append(lev)
                    tck_locs[direction].append(t)
        return gi

    def set_transform(self, aux_trans):
        if isinstance(aux_trans, Transform):
            self._aux_transform = aux_trans
        elif len(aux_trans) == 2 and all(map(callable, aux_trans)):
            self._aux_transform = _User2DTransform(*aux_trans)
        else:
            raise TypeError("'aux_trans' must be either a Transform instance or a pair of callables")

    def get_transform(self):
        return self._aux_transform
    update_transform = set_transform

    def transform_xy(self, x, y):
        return self._aux_transform.transform(np.column_stack([x, y])).T

    def inv_transform_xy(self, x, y):
        return self._aux_transform.inverted().transform(np.column_stack([x, y])).T

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in ['extreme_finder', 'grid_locator1', 'grid_locator2', 'tick_formatter1', 'tick_formatter2']:
                setattr(self, k, v)
            else:
                raise ValueError(f'Unknown update property {k!r}')