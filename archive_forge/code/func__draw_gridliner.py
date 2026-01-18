import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _draw_gridliner(self, nx=None, ny=None, renderer=None):
    """Create Artists for all visible elements and add to our Axes.

        The following rules apply for the visibility of labels:

        - X-type labels are plotted along the bottom, top and geo spines.
        - Y-type labels are plotted along the left, right and geo spines.
        - A label must not overlap another label marked as visible.
        - A label must not overlap the map boundary.
        - When a label is about to be hidden, its padding is slightly
          increase until it can be drawn or until a padding limit is reached.
        """
    if self._drawn and (not self._auto_update):
        return
    self._drawn = True
    lon_lim, lat_lim = self._axes_domain(nx=nx, ny=ny)
    transform = self._crs_transform()
    n_steps = self.n_steps
    crs = self.crs
    lon_ticks = self.xlocator.tick_values(lon_lim[0], lon_lim[1])
    lat_ticks = self.ylocator.tick_values(lat_lim[0], lat_lim[1])
    inf = max(lon_lim[0], crs.x_limits[0])
    sup = min(lon_lim[1], crs.x_limits[1])
    lon_ticks = [value for value in lon_ticks if inf <= value <= sup]
    inf = max(lat_lim[0], crs.y_limits[0])
    sup = min(lat_lim[1], crs.y_limits[1])
    lat_ticks = [value for value in lat_ticks if inf <= value <= sup]
    collection_kwargs = self.collection_kwargs
    if collection_kwargs is None:
        collection_kwargs = {}
    collection_kwargs = collection_kwargs.copy()
    collection_kwargs['transform'] = transform
    if not any((x in collection_kwargs for x in ['c', 'color'])):
        collection_kwargs.setdefault('color', matplotlib.rcParams['grid.color'])
    if not any((x in collection_kwargs for x in ['ls', 'linestyle'])):
        collection_kwargs.setdefault('linestyle', matplotlib.rcParams['grid.linestyle'])
    if not any((x in collection_kwargs for x in ['lw', 'linewidth'])):
        collection_kwargs.setdefault('linewidth', matplotlib.rcParams['grid.linewidth'])
    collection_kwargs.setdefault('clip_path', self.axes.patch)
    lat_min, lat_max = lat_lim
    if lat_ticks:
        lat_min = min(lat_min, min(lat_ticks))
        lat_max = max(lat_max, max(lat_ticks))
    lon_lines = np.empty((len(lon_ticks), n_steps, 2))
    lon_lines[:, :, 0] = np.array(lon_ticks)[:, np.newaxis]
    lon_lines[:, :, 1] = np.linspace(lat_min, lat_max, n_steps)[np.newaxis, :]
    if self.xlines:
        nx = len(lon_lines) + 1
        if isinstance(crs, Projection) and isinstance(crs, _RectangularProjection) and (abs(np.diff(lon_lim)) == abs(np.diff(crs.x_limits))):
            nx -= 1
        if self.xline_artists:
            lon_lc, = self.xline_artists
            lon_lc.set(segments=lon_lines, **collection_kwargs)
        else:
            lon_lc = mcollections.LineCollection(lon_lines, **collection_kwargs)
            self.xline_artists.append(lon_lc)
    lon_min, lon_max = lon_lim
    if lon_ticks:
        lon_min = min(lon_min, min(lon_ticks))
        lon_max = max(lon_max, max(lon_ticks))
    lat_lines = np.empty((len(lat_ticks), n_steps, 2))
    lat_lines[:, :, 0] = np.linspace(lon_min, lon_max, n_steps)[np.newaxis, :]
    lat_lines[:, :, 1] = np.array(lat_ticks)[:, np.newaxis]
    if self.ylines:
        if self.yline_artists:
            lat_lc, = self.yline_artists
            lat_lc.set(segments=lat_lines, **collection_kwargs)
        else:
            lat_lc = mcollections.LineCollection(lat_lines, **collection_kwargs)
            self.yline_artists.append(lat_lc)
    self._labels.clear()
    if not any((self.left_labels, self.right_labels, self.bottom_labels, self.top_labels, self.inline_labels, self.geo_labels)):
        return
    self._assert_can_draw_ticks()
    max_padding_factor = 5
    delta_padding_factor = 0.2
    spines_specs = {'left': {'index': 0, 'coord_type': 'x', 'opcmp': operator.le, 'opval': max}, 'bottom': {'index': 1, 'coord_type': 'y', 'opcmp': operator.le, 'opval': max}, 'right': {'index': 0, 'coord_type': 'x', 'opcmp': operator.ge, 'opval': min}, 'top': {'index': 1, 'coord_type': 'y', 'opcmp': operator.ge, 'opval': min}}
    for side, specs in spines_specs.items():
        bbox = self.axes.spines[side].get_window_extent(renderer)
        specs['coords'] = [getattr(bbox, specs['coord_type'] + idx) for idx in '01']

    def update_artist(artist, renderer):
        artist.update_bbox_position_size(renderer)
        this_patch = artist.get_bbox_patch()
        this_path = this_patch.get_path().transformed(this_patch.get_transform())
        return this_path
    self.axes.spines['geo'].get_window_extent(renderer)
    map_boundary_path = self.axes.spines['geo'].get_path().transformed(self.axes.spines['geo'].get_transform())
    map_boundary = sgeom.Polygon(map_boundary_path.vertices)
    if self.x_inline:
        y_midpoints = self._find_midpoints(lat_lim, lat_ticks)
    if self.y_inline:
        x_midpoints = self._find_midpoints(lon_lim, lon_ticks)
    crs_transform = self._crs_transform().transform
    inverse_data_transform = self.axes.transData.inverted().transform_point
    generate_labels = self._generate_labels()
    for xylabel, lines, line_ticks, formatter, label_style in (('x', lon_lines, lon_ticks, self.xformatter, self.xlabel_style.copy()), ('y', lat_lines, lat_ticks, self.yformatter, self.ylabel_style.copy())):
        x_inline = self.x_inline and xylabel == 'x'
        y_inline = self.y_inline and xylabel == 'y'
        padding = getattr(self, f'{xylabel}padding')
        bbox_style = self.labels_bbox_style.copy()
        if 'bbox' in label_style:
            bbox_style.update(label_style['bbox'])
        label_style['bbox'] = bbox_style
        formatter.set_locs(line_ticks)
        for line_coords, tick_value in zip(lines, line_ticks):
            line_coords = crs_transform(line_coords)
            infs = np.isnan(line_coords).any(axis=1)
            line_coords = line_coords.compress(~infs, axis=0)
            if line_coords.size == 0:
                continue
            line = sgeom.LineString(line_coords)
            if not line.intersects(map_boundary):
                continue
            intersection = line.intersection(map_boundary)
            del line
            if intersection.is_empty:
                continue
            if isinstance(intersection, sgeom.MultiPoint):
                if len(intersection) < 2:
                    continue
                n2 = min(len(intersection), 3)
                tails = [[(pt.x, pt.y) for pt in intersection[:n2:n2 - 1]]]
                heads = [[(pt.x, pt.y) for pt in intersection[-1:-n2 - 1:-n2 + 1]]]
            elif isinstance(intersection, (sgeom.LineString, sgeom.MultiLineString)):
                if isinstance(intersection, sgeom.LineString):
                    intersection = [intersection]
                elif len(intersection.geoms) > 4:
                    xy = np.append(intersection.geoms[0].coords, intersection.geoms[-1].coords, axis=0)
                    intersection = [sgeom.LineString(xy)]
                else:
                    intersection = intersection.geoms
                tails = []
                heads = []
                for inter in intersection:
                    if len(inter.coords) < 2:
                        continue
                    n2 = min(len(inter.coords), 8)
                    tails.append(inter.coords[:n2:n2 - 1])
                    heads.append(inter.coords[-1:-n2 - 1:-n2 + 1])
                if not tails:
                    continue
            elif isinstance(intersection, sgeom.GeometryCollection):
                xy = []
                for geom in intersection.geoms:
                    for coord in geom.coords:
                        xy.append(coord)
                        if len(xy) == 2:
                            break
                    if len(xy) == 2:
                        break
                tails = [xy]
                xy = []
                for geom in reversed(intersection.geoms):
                    for coord in reversed(geom.coords):
                        xy.append(coord)
                        if len(xy) == 2:
                            break
                    if len(xy) == 2:
                        break
                heads = [xy]
            else:
                warnings.warn(f'Unsupported intersection geometry for gridline labels: {intersection.__class__.__name__}')
                continue
            del intersection
            for i, (pt0, pt1) in itertools.chain.from_iterable((enumerate(pair) for pair in zip(tails, heads))):
                x0, y0 = pt0
                if x_inline or y_inline:
                    kw = {'rotation': 0, 'transform': self.crs, 'ha': 'center', 'va': 'center'}
                    loc = 'inline'
                else:
                    x1, y1 = pt1
                    segment_angle = np.arctan2(y0 - y1, x0 - x1) * 180 / np.pi
                    loc = self._get_loc_from_spine_intersection(spines_specs, xylabel, x0, y0)
                    if not self._draw_this_label(xylabel, loc):
                        visible = False
                    kw = self._get_text_specs(segment_angle, loc, xylabel)
                    kw['transform'] = self._get_padding_transform(segment_angle, loc, xylabel)
                kw.update(label_style)
                pt0 = inverse_data_transform(pt0)
                if y_inline:
                    if abs(tick_value) == 180:
                        continue
                    x = x_midpoints[i]
                    y = tick_value
                    kw.update(clip_on=True)
                    y_set = True
                else:
                    x = pt0[0]
                    y_set = False
                if x_inline:
                    if abs(tick_value) == 180:
                        continue
                    x = tick_value
                    y = y_midpoints[i]
                    kw.update(clip_on=True)
                elif not y_set:
                    y = pt0[1]
                label = next(generate_labels)
                text = formatter(tick_value)
                artist = label.artist
                artist.set(x=x, y=y, text=text, **kw)
                this_path = update_artist(artist, renderer)
                if not x_inline and (not y_inline) and (loc == 'geo'):
                    new_loc = self._get_loc_from_spine_overlapping(spines_specs, xylabel, this_path)
                    if new_loc and loc != new_loc:
                        loc = new_loc
                        transform = self._get_padding_transform(segment_angle, loc, xylabel)
                        artist.set_transform(transform)
                        artist.update(self._get_text_specs(segment_angle, loc, xylabel))
                        artist.update(label_style.copy())
                        this_path = update_artist(artist, renderer)
                if not self._draw_this_label(xylabel, loc):
                    visible = False
                elif x_inline or y_inline:
                    center = artist.get_transform().transform_point(artist.get_position())
                    visible = map_boundary_path.contains_point(center)
                else:
                    visible = False
                    padding_factor = 1
                    while padding_factor < max_padding_factor:
                        if map_boundary_path.intersects_path(this_path, filled=padding > 0):
                            transform = self._get_padding_transform(segment_angle, loc, xylabel, padding_factor)
                            artist.set_transform(transform)
                            this_path = update_artist(artist, renderer)
                            padding_factor += delta_padding_factor
                        else:
                            visible = True
                            break
                label.set_visible(visible)
                label.path = this_path
                label.xy = xylabel
                label.loc = loc
                self._labels.append(label)
    if self._labels:
        self._labels.sort(key=operator.attrgetter('priority'), reverse=True)
        visible_labels = []
        for label in self._labels:
            if label.get_visible():
                for other_label in visible_labels:
                    if label.check_overlapping(other_label):
                        break
                else:
                    visible_labels.append(label)