from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
class _DistributionPlotter(VectorPlotter):
    wide_structure = {'x': '@values', 'hue': '@columns'}
    flat_structure = {'x': '@values'}

    def __init__(self, data=None, variables={}):
        super().__init__(data=data, variables=variables)

    @property
    def univariate(self):
        """Return True if only x or y are used."""
        return bool({'x', 'y'} - set(self.variables))

    @property
    def data_variable(self):
        """Return the variable with data for univariate plots."""
        if not self.univariate:
            raise AttributeError('This is not a univariate plot')
        return {'x', 'y'}.intersection(self.variables).pop()

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        return bool({'x', 'y'} & set(self.variables))

    def _add_legend(self, ax_obj, artist, fill, element, multiple, alpha, artist_kws, legend_kws):
        """Add artists that reflect semantic mappings and put then in a legend."""
        handles = []
        labels = []
        for level in self._hue_map.levels:
            color = self._hue_map(level)
            kws = self._artist_kws(artist_kws, fill, element, multiple, color, alpha)
            if 'facecolor' in kws:
                kws.pop('color', None)
            handles.append(artist(**kws))
            labels.append(level)
        if isinstance(ax_obj, mpl.axes.Axes):
            ax_obj.legend(handles, labels, title=self.variables['hue'], **legend_kws)
        else:
            legend_data = dict(zip(labels, handles))
            ax_obj.add_legend(legend_data, title=self.variables['hue'], label_order=self.var_levels['hue'], **legend_kws)

    def _artist_kws(self, kws, fill, element, multiple, color, alpha):
        """Handle differences between artists in filled/unfilled plots."""
        kws = kws.copy()
        if fill:
            kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
            kws.setdefault('facecolor', to_rgba(color, alpha))
            if element == 'bars':
                kws['color'] = 'none'
            if multiple in ['stack', 'fill'] or element == 'bars':
                kws.setdefault('edgecolor', mpl.rcParams['patch.edgecolor'])
            else:
                kws.setdefault('edgecolor', to_rgba(color, 1))
        elif element == 'bars':
            kws['facecolor'] = 'none'
            kws['edgecolor'] = to_rgba(color, alpha)
        else:
            kws['color'] = to_rgba(color, alpha)
        return kws

    def _quantile_to_level(self, data, quantile):
        """Return data levels corresponding to quantile cuts of mass."""
        isoprop = np.asarray(quantile)
        values = np.ravel(data)
        sorted_values = np.sort(values)[::-1]
        normalized_values = np.cumsum(sorted_values) / values.sum()
        idx = np.searchsorted(normalized_values, 1 - isoprop)
        levels = np.take(sorted_values, idx, mode='clip')
        return levels

    def _cmap_from_color(self, color):
        """Return a sequential colormap given a color seed."""
        r, g, b, _ = to_rgba(color)
        h, s, _ = husl.rgb_to_husl(r, g, b)
        xx = np.linspace(-1, 1, int(1.15 * 256))[:256]
        ramp = np.zeros((256, 3))
        ramp[:, 0] = h
        ramp[:, 1] = s * np.cos(xx)
        ramp[:, 2] = np.linspace(35, 80, 256)
        colors = np.clip([husl.husl_to_rgb(*hsl) for hsl in ramp], 0, 1)
        return mpl.colors.ListedColormap(colors[::-1])

    def _default_discrete(self):
        """Find default values for discrete hist estimation based on variable type."""
        if self.univariate:
            discrete = self.var_types[self.data_variable] == 'categorical'
        else:
            discrete_x = self.var_types['x'] == 'categorical'
            discrete_y = self.var_types['y'] == 'categorical'
            discrete = (discrete_x, discrete_y)
        return discrete

    def _resolve_multiple(self, curves, multiple):
        """Modify the density data structure to handle multiple densities."""
        baselines = {k: np.zeros_like(v) for k, v in curves.items()}
        if 'hue' not in self.variables:
            return (curves, baselines)
        if multiple in ('stack', 'fill'):
            curves = pd.DataFrame(curves).iloc[:, ::-1]
            column_groups = {}
            for i, keyd in enumerate(map(dict, curves.columns)):
                facet_key = (keyd.get('col', None), keyd.get('row', None))
                column_groups.setdefault(facet_key, [])
                column_groups[facet_key].append(i)
            baselines = curves.copy()
            for col_idxs in column_groups.values():
                cols = curves.columns[col_idxs]
                norm_constant = curves[cols].sum(axis='columns')
                curves[cols] = curves[cols].cumsum(axis='columns')
                if multiple == 'fill':
                    curves[cols] = curves[cols].div(norm_constant, axis='index')
                baselines[cols] = curves[cols].shift(1, axis=1).fillna(0)
        if multiple == 'dodge':
            hue_levels = self.var_levels['hue']
            n = len(hue_levels)
            f_fwd, f_inv = self._get_scale_transforms(self.data_variable)
            for key in curves:
                level = dict(key)['hue']
                hist = curves[key].reset_index(name='heights')
                level_idx = hue_levels.index(level)
                a = f_fwd(hist['edges'])
                b = f_fwd(hist['edges'] + hist['widths'])
                w = (b - a) / n
                new_min = f_inv(a + level_idx * w)
                new_max = f_inv(a + (level_idx + 1) * w)
                hist['widths'] = new_max - new_min
                hist['edges'] = new_min
                curves[key] = hist.set_index(['edges', 'widths'])['heights']
        return (curves, baselines)

    def _compute_univariate_density(self, data_variable, common_norm, common_grid, estimate_kws, warn_singular=True):
        estimator = KDE(**estimate_kws)
        if set(self.variables) - {'x', 'y'}:
            if common_grid:
                all_observations = self.comp_data.dropna()
                estimator.define_support(all_observations[data_variable])
        else:
            common_norm = False
        all_data = self.plot_data.dropna()
        if common_norm and 'weights' in all_data:
            whole_weight = all_data['weights'].sum()
        else:
            whole_weight = len(all_data)
        densities = {}
        for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
            observations = sub_data[data_variable]
            if 'weights' in self.variables:
                weights = sub_data['weights']
                part_weight = weights.sum()
            else:
                weights = None
                part_weight = len(sub_data)
            variance = np.nan_to_num(observations.var())
            singular = len(observations) < 2 or math.isclose(variance, 0)
            try:
                if not singular:
                    density, support = estimator(observations, weights=weights)
            except np.linalg.LinAlgError:
                singular = True
            if singular:
                msg = 'Dataset has 0 variance; skipping density estimate. Pass `warn_singular=False` to disable this warning.'
                if warn_singular:
                    warnings.warn(msg, UserWarning, stacklevel=4)
                continue
            _, f_inv = self._get_scale_transforms(self.data_variable)
            support = f_inv(support)
            if common_norm:
                density *= part_weight / whole_weight
            key = tuple(sub_vars.items())
            densities[key] = pd.Series(density, index=support)
        return densities

    def plot_univariate_histogram(self, multiple, element, fill, common_norm, common_bins, shrink, kde, kde_kws, color, legend, line_kws, estimate_kws, **plot_kws):
        kde_kws = {} if kde_kws is None else kde_kws.copy()
        line_kws = {} if line_kws is None else line_kws.copy()
        estimate_kws = {} if estimate_kws is None else estimate_kws.copy()
        _check_argument('multiple', ['layer', 'stack', 'fill', 'dodge'], multiple)
        _check_argument('element', ['bars', 'step', 'poly'], element)
        auto_bins_with_weights = 'weights' in self.variables and estimate_kws['bins'] == 'auto' and (estimate_kws['binwidth'] is None) and (not estimate_kws['discrete'])
        if auto_bins_with_weights:
            msg = "`bins` cannot be 'auto' when using weights. Setting `bins=10`, but you will likely want to adjust."
            warnings.warn(msg, UserWarning)
            estimate_kws['bins'] = 10
        if estimate_kws['stat'] == 'count':
            common_norm = False
        orient = self.data_variable
        estimator = Hist(**estimate_kws)
        histograms = {}
        all_data = self.comp_data.dropna()
        all_weights = all_data.get('weights', None)
        multiple_histograms = set(self.variables) - {'x', 'y'}
        if multiple_histograms:
            if common_bins:
                bin_kws = estimator._define_bin_params(all_data, orient, None)
        else:
            common_norm = False
        if common_norm and all_weights is not None:
            whole_weight = all_weights.sum()
        else:
            whole_weight = len(all_data)
        if kde:
            kde_kws.setdefault('cut', 0)
            kde_kws['cumulative'] = estimate_kws['cumulative']
            densities = self._compute_univariate_density(self.data_variable, common_norm, common_bins, kde_kws, warn_singular=False)
        for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
            key = tuple(sub_vars.items())
            orient = self.data_variable
            if 'weights' in self.variables:
                sub_data['weight'] = sub_data.pop('weights')
                part_weight = sub_data['weight'].sum()
            else:
                part_weight = len(sub_data)
            if not (multiple_histograms and common_bins):
                bin_kws = estimator._define_bin_params(sub_data, orient, None)
            res = estimator._normalize(estimator._eval(sub_data, orient, bin_kws))
            heights = res[estimator.stat].to_numpy()
            widths = res['space'].to_numpy()
            edges = res[orient].to_numpy() - widths / 2
            if kde and key in densities:
                density = densities[key]
                if estimator.cumulative:
                    hist_norm = heights.max()
                else:
                    hist_norm = (heights * widths).sum()
                densities[key] *= hist_norm
            ax = self._get_axes(sub_vars)
            _, inv = _get_transform_functions(ax, self.data_variable)
            widths = inv(edges + widths) - inv(edges)
            edges = inv(edges)
            edges = edges + (1 - shrink) / 2 * widths
            widths *= shrink
            index = pd.MultiIndex.from_arrays([pd.Index(edges, name='edges'), pd.Index(widths, name='widths')])
            hist = pd.Series(heights, index=index, name='heights')
            if common_norm:
                hist *= part_weight / whole_weight
            histograms[key] = hist
        histograms, baselines = self._resolve_multiple(histograms, multiple)
        if kde:
            densities, _ = self._resolve_multiple(densities, None if multiple == 'dodge' else multiple)
        sticky_stat = (0, 1) if multiple == 'fill' else (0, np.inf)
        if multiple == 'fill':
            bin_vals = histograms.index.to_frame()
            edges = bin_vals['edges']
            widths = bin_vals['widths']
            sticky_data = (edges.min(), edges.max() + widths.loc[edges.idxmax()])
        else:
            sticky_data = []
        if fill:
            if 'hue' in self.variables and multiple == 'layer':
                default_alpha = 0.5 if element == 'bars' else 0.25
            elif kde:
                default_alpha = 0.5
            else:
                default_alpha = 0.75
        else:
            default_alpha = 1
        alpha = plot_kws.pop('alpha', default_alpha)
        hist_artists = []
        for sub_vars, _ in self.iter_data('hue', reverse=True):
            key = tuple(sub_vars.items())
            hist = histograms[key].rename('heights').reset_index()
            bottom = np.asarray(baselines[key])
            ax = self._get_axes(sub_vars)
            if 'hue' in self.variables:
                sub_color = self._hue_map(sub_vars['hue'])
            else:
                sub_color = color
            artist_kws = self._artist_kws(plot_kws, fill, element, multiple, sub_color, alpha)
            if element == 'bars':
                plot_func = ax.bar if self.data_variable == 'x' else ax.barh
                artists = plot_func(hist['edges'], hist['heights'] - bottom, hist['widths'], bottom, align='edge', **artist_kws)
                for bar in artists:
                    if self.data_variable == 'x':
                        bar.sticky_edges.x[:] = sticky_data
                        bar.sticky_edges.y[:] = sticky_stat
                    else:
                        bar.sticky_edges.x[:] = sticky_stat
                        bar.sticky_edges.y[:] = sticky_data
                hist_artists.extend(artists)
            else:
                if element == 'step':
                    final = hist.iloc[-1]
                    x = np.append(hist['edges'], final['edges'] + final['widths'])
                    y = np.append(hist['heights'], final['heights'])
                    b = np.append(bottom, bottom[-1])
                    if self.data_variable == 'x':
                        step = 'post'
                        drawstyle = 'steps-post'
                    else:
                        step = 'post'
                        drawstyle = 'steps-pre'
                elif element == 'poly':
                    x = hist['edges'] + hist['widths'] / 2
                    y = hist['heights']
                    b = bottom
                    step = None
                    drawstyle = None
                if self.data_variable == 'x':
                    if fill:
                        artist = ax.fill_between(x, b, y, step=step, **artist_kws)
                    else:
                        artist, = ax.plot(x, y, drawstyle=drawstyle, **artist_kws)
                    artist.sticky_edges.x[:] = sticky_data
                    artist.sticky_edges.y[:] = sticky_stat
                else:
                    if fill:
                        artist = ax.fill_betweenx(x, b, y, step=step, **artist_kws)
                    else:
                        artist, = ax.plot(y, x, drawstyle=drawstyle, **artist_kws)
                    artist.sticky_edges.x[:] = sticky_stat
                    artist.sticky_edges.y[:] = sticky_data
                hist_artists.append(artist)
            if kde:
                try:
                    density = densities[key]
                except KeyError:
                    continue
                support = density.index
                if 'x' in self.variables:
                    line_args = (support, density)
                    sticky_x, sticky_y = (None, (0, np.inf))
                else:
                    line_args = (density, support)
                    sticky_x, sticky_y = ((0, np.inf), None)
                line_kws['color'] = to_rgba(sub_color, 1)
                line, = ax.plot(*line_args, **line_kws)
                if sticky_x is not None:
                    line.sticky_edges.x[:] = sticky_x
                if sticky_y is not None:
                    line.sticky_edges.y[:] = sticky_y
        if element == 'bars' and 'linewidth' not in plot_kws:
            hist_metadata = pd.concat([h.index.to_frame() for _, h in histograms.items()]).reset_index(drop=True)
            thin_bar_idx = hist_metadata['widths'].idxmin()
            binwidth = hist_metadata.loc[thin_bar_idx, 'widths']
            left_edge = hist_metadata.loc[thin_bar_idx, 'edges']
            default_linewidth = math.inf
            for sub_vars, _ in self.iter_data():
                ax = self._get_axes(sub_vars)
                ax.autoscale_view()
                pts_x, pts_y = 72 / ax.figure.dpi * abs(ax.transData.transform([left_edge + binwidth] * 2) - ax.transData.transform([left_edge] * 2))
                if self.data_variable == 'x':
                    binwidth_points = pts_x
                else:
                    binwidth_points = pts_y
                default_linewidth = min(0.1 * binwidth_points, default_linewidth)
            for bar in hist_artists:
                max_linewidth = bar.get_linewidth()
                if not fill:
                    max_linewidth *= 1.5
                linewidth = min(default_linewidth, max_linewidth)
                if not fill:
                    min_linewidth = 0.5
                    linewidth = max(linewidth, min_linewidth)
                bar.set_linewidth(linewidth)
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        default_x = default_y = ''
        if self.data_variable == 'x':
            default_y = estimator.stat.capitalize()
        if self.data_variable == 'y':
            default_x = estimator.stat.capitalize()
        self._add_axis_labels(ax, default_x, default_y)
        if 'hue' in self.variables and legend:
            if fill or element == 'bars':
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(ax_obj, artist, fill, element, multiple, alpha, plot_kws, {})

    def plot_bivariate_histogram(self, common_bins, common_norm, thresh, pthresh, pmax, color, legend, cbar, cbar_ax, cbar_kws, estimate_kws, **plot_kws):
        cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
        estimator = Histogram(**estimate_kws)
        if set(self.variables) - {'x', 'y'}:
            all_data = self.comp_data.dropna()
            if common_bins:
                estimator.define_bin_params(all_data['x'], all_data['y'], all_data.get('weights', None))
        else:
            common_norm = False
        full_heights = []
        for _, sub_data in self.iter_data(from_comp_data=True):
            sub_heights, _ = estimator(sub_data['x'], sub_data['y'], sub_data.get('weights', None))
            full_heights.append(sub_heights)
        common_color_norm = not set(self.variables) - {'x', 'y'} or common_norm
        if pthresh is not None and common_color_norm:
            thresh = self._quantile_to_level(full_heights, pthresh)
        plot_kws.setdefault('vmin', 0)
        if common_color_norm:
            if pmax is not None:
                vmax = self._quantile_to_level(full_heights, pmax)
            else:
                vmax = plot_kws.pop('vmax', max(map(np.max, full_heights)))
        else:
            vmax = None
        if color is None:
            color = 'C0'
        for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
            if sub_data.empty:
                continue
            heights, (x_edges, y_edges) = estimator(sub_data['x'], sub_data['y'], weights=sub_data.get('weights', None))
            ax = self._get_axes(sub_vars)
            _, inv_x = _get_transform_functions(ax, 'x')
            _, inv_y = _get_transform_functions(ax, 'y')
            x_edges = inv_x(x_edges)
            y_edges = inv_y(y_edges)
            if estimator.stat != 'count' and common_norm:
                heights *= len(sub_data) / len(all_data)
            artist_kws = plot_kws.copy()
            if 'hue' in self.variables:
                color = self._hue_map(sub_vars['hue'])
                cmap = self._cmap_from_color(color)
                artist_kws['cmap'] = cmap
            else:
                cmap = artist_kws.pop('cmap', None)
                if isinstance(cmap, str):
                    cmap = color_palette(cmap, as_cmap=True)
                elif cmap is None:
                    cmap = self._cmap_from_color(color)
                artist_kws['cmap'] = cmap
            if not common_color_norm and pmax is not None:
                vmax = self._quantile_to_level(heights, pmax)
            if vmax is not None:
                artist_kws['vmax'] = vmax
            if not common_color_norm and pthresh:
                thresh = self._quantile_to_level(heights, pthresh)
            if thresh is not None:
                heights = np.ma.masked_less_equal(heights, thresh)
            x_grid = any([l.get_visible() for l in ax.xaxis.get_gridlines()])
            y_grid = any([l.get_visible() for l in ax.yaxis.get_gridlines()])
            mesh = ax.pcolormesh(x_edges, y_edges, heights.T, **artist_kws)
            if thresh is not None:
                mesh.sticky_edges.x[:] = []
                mesh.sticky_edges.y[:] = []
            if cbar:
                ax.figure.colorbar(mesh, cbar_ax, ax, **cbar_kws)
            if x_grid:
                ax.grid(True, axis='x')
            if y_grid:
                ax.grid(True, axis='y')
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        self._add_axis_labels(ax)
        if 'hue' in self.variables and legend:
            artist_kws = {}
            artist = partial(mpl.patches.Patch)
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(ax_obj, artist, True, False, 'layer', 1, artist_kws, {})

    def plot_univariate_density(self, multiple, common_norm, common_grid, warn_singular, fill, color, legend, estimate_kws, **plot_kws):
        if fill is None:
            fill = multiple in ('stack', 'fill')
        if fill:
            artist = mpl.collections.PolyCollection
        else:
            artist = mpl.lines.Line2D
        plot_kws = normalize_kwargs(plot_kws, artist)
        _check_argument('multiple', ['layer', 'stack', 'fill'], multiple)
        subsets = bool(set(self.variables) - {'x', 'y'})
        if subsets and multiple in ('stack', 'fill'):
            common_grid = True
        densities = self._compute_univariate_density(self.data_variable, common_norm, common_grid, estimate_kws, warn_singular)
        densities, baselines = self._resolve_multiple(densities, multiple)
        sticky_density = (0, 1) if multiple == 'fill' else (0, np.inf)
        if multiple == 'fill':
            sticky_support = (densities.index.min(), densities.index.max())
        else:
            sticky_support = []
        if fill:
            if multiple == 'layer':
                default_alpha = 0.25
            else:
                default_alpha = 0.75
        else:
            default_alpha = 1
        alpha = plot_kws.pop('alpha', default_alpha)
        for sub_vars, _ in self.iter_data('hue', reverse=True):
            key = tuple(sub_vars.items())
            try:
                density = densities[key]
            except KeyError:
                continue
            support = density.index
            fill_from = baselines[key]
            ax = self._get_axes(sub_vars)
            if 'hue' in self.variables:
                sub_color = self._hue_map(sub_vars['hue'])
            else:
                sub_color = color
            artist_kws = self._artist_kws(plot_kws, fill, False, multiple, sub_color, alpha)
            if 'x' in self.variables:
                if fill:
                    artist = ax.fill_between(support, fill_from, density, **artist_kws)
                else:
                    artist, = ax.plot(support, density, **artist_kws)
                artist.sticky_edges.x[:] = sticky_support
                artist.sticky_edges.y[:] = sticky_density
            else:
                if fill:
                    artist = ax.fill_betweenx(support, fill_from, density, **artist_kws)
                else:
                    artist, = ax.plot(density, support, **artist_kws)
                artist.sticky_edges.x[:] = sticky_density
                artist.sticky_edges.y[:] = sticky_support
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        default_x = default_y = ''
        if self.data_variable == 'x':
            default_y = 'Density'
        if self.data_variable == 'y':
            default_x = 'Density'
        self._add_axis_labels(ax, default_x, default_y)
        if 'hue' in self.variables and legend:
            if fill:
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(ax_obj, artist, fill, False, multiple, alpha, plot_kws, {})

    def plot_bivariate_density(self, common_norm, fill, levels, thresh, color, legend, cbar, warn_singular, cbar_ax, cbar_kws, estimate_kws, **contour_kws):
        contour_kws = contour_kws.copy()
        estimator = KDE(**estimate_kws)
        if not set(self.variables) - {'x', 'y'}:
            common_norm = False
        all_data = self.plot_data.dropna()
        densities, supports = ({}, {})
        for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
            observations = sub_data[['x', 'y']]
            min_variance = observations.var().fillna(0).min()
            observations = (observations['x'], observations['y'])
            if 'weights' in self.variables:
                weights = sub_data['weights']
            else:
                weights = None
            singular = math.isclose(min_variance, 0)
            try:
                if not singular:
                    density, support = estimator(*observations, weights=weights)
            except np.linalg.LinAlgError:
                singular = True
            if singular:
                msg = 'KDE cannot be estimated (0 variance or perfect covariance). Pass `warn_singular=False` to disable this warning.'
                if warn_singular:
                    warnings.warn(msg, UserWarning, stacklevel=3)
                continue
            ax = self._get_axes(sub_vars)
            _, inv_x = _get_transform_functions(ax, 'x')
            _, inv_y = _get_transform_functions(ax, 'y')
            support = (inv_x(support[0]), inv_y(support[1]))
            if common_norm:
                density *= len(sub_data) / len(all_data)
            key = tuple(sub_vars.items())
            densities[key] = density
            supports[key] = support
        if thresh is None:
            thresh = 0
        if isinstance(levels, Number):
            levels = np.linspace(thresh, 1, levels)
        elif min(levels) < 0 or max(levels) > 1:
            raise ValueError('levels must be in [0, 1]')
        if common_norm:
            common_levels = self._quantile_to_level(list(densities.values()), levels)
            draw_levels = {k: common_levels for k in densities}
        else:
            draw_levels = {k: self._quantile_to_level(d, levels) for k, d in densities.items()}
        if 'hue' in self.variables:
            for param in ['cmap', 'colors']:
                if param in contour_kws:
                    msg = f'{param} parameter ignored when using hue mapping.'
                    warnings.warn(msg, UserWarning)
                    contour_kws.pop(param)
        else:
            coloring_given = set(contour_kws) & {'cmap', 'colors'}
            if fill and (not coloring_given):
                cmap = self._cmap_from_color(color)
                contour_kws['cmap'] = cmap
            if not fill and (not coloring_given):
                contour_kws['colors'] = [color]
            cmap = contour_kws.pop('cmap', None)
            if isinstance(cmap, str):
                cmap = color_palette(cmap, as_cmap=True)
            if cmap is not None:
                contour_kws['cmap'] = cmap
        for sub_vars, _ in self.iter_data('hue'):
            if 'hue' in sub_vars:
                color = self._hue_map(sub_vars['hue'])
                if fill:
                    contour_kws['cmap'] = self._cmap_from_color(color)
                else:
                    contour_kws['colors'] = [color]
            ax = self._get_axes(sub_vars)
            if fill:
                contour_func = ax.contourf
            else:
                contour_func = ax.contour
            key = tuple(sub_vars.items())
            if key not in densities:
                continue
            density = densities[key]
            xx, yy = supports[key]
            contour_kws.pop('label', None)
            cset = contour_func(xx, yy, density, levels=draw_levels[key], **contour_kws)
            if cbar:
                cbar_kws = {} if cbar_kws is None else cbar_kws
                ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        self._add_axis_labels(ax)
        if 'hue' in self.variables and legend:
            artist_kws = {}
            if fill:
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(ax_obj, artist, fill, False, 'layer', 1, artist_kws, {})

    def plot_univariate_ecdf(self, estimate_kws, legend, **plot_kws):
        estimator = ECDF(**estimate_kws)
        drawstyles = dict(x='steps-post', y='steps-pre')
        plot_kws['drawstyle'] = drawstyles[self.data_variable]
        for sub_vars, sub_data in self.iter_data('hue', reverse=True, from_comp_data=True):
            if sub_data.empty:
                continue
            observations = sub_data[self.data_variable]
            weights = sub_data.get('weights', None)
            stat, vals = estimator(observations, weights=weights)
            artist_kws = plot_kws.copy()
            if 'hue' in self.variables:
                artist_kws['color'] = self._hue_map(sub_vars['hue'])
            ax = self._get_axes(sub_vars)
            _, inv = _get_transform_functions(ax, self.data_variable)
            vals = inv(vals)
            if isinstance(inv.__self__, mpl.scale.LogTransform):
                vals[0] = -np.inf
            if self.data_variable == 'x':
                plot_args = (vals, stat)
                stat_variable = 'y'
            else:
                plot_args = (stat, vals)
                stat_variable = 'x'
            if estimator.stat == 'count':
                top_edge = len(observations)
            else:
                top_edge = 1
            artist, = ax.plot(*plot_args, **artist_kws)
            sticky_edges = getattr(artist.sticky_edges, stat_variable)
            sticky_edges[:] = (0, top_edge)
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        stat = estimator.stat.capitalize()
        default_x = default_y = ''
        if self.data_variable == 'x':
            default_y = stat
        if self.data_variable == 'y':
            default_x = stat
        self._add_axis_labels(ax, default_x, default_y)
        if 'hue' in self.variables and legend:
            artist = partial(mpl.lines.Line2D, [], [])
            alpha = plot_kws.get('alpha', 1)
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(ax_obj, artist, False, False, None, alpha, plot_kws, {})

    def plot_rug(self, height, expand_margins, legend, **kws):
        for sub_vars, sub_data in self.iter_data(from_comp_data=True):
            ax = self._get_axes(sub_vars)
            kws.setdefault('linewidth', 1)
            if expand_margins:
                xmarg, ymarg = ax.margins()
                if 'x' in self.variables:
                    ymarg += height * 2
                if 'y' in self.variables:
                    xmarg += height * 2
                ax.margins(x=xmarg, y=ymarg)
            if 'hue' in self.variables:
                kws.pop('c', None)
                kws.pop('color', None)
            if 'x' in self.variables:
                self._plot_single_rug(sub_data, 'x', height, ax, kws)
            if 'y' in self.variables:
                self._plot_single_rug(sub_data, 'y', height, ax, kws)
            self._add_axis_labels(ax)
            if 'hue' in self.variables and legend:
                legend_artist = partial(mpl.lines.Line2D, [], [])
                self._add_legend(ax, legend_artist, False, False, None, 1, {}, {})

    def _plot_single_rug(self, sub_data, var, height, ax, kws):
        """Draw a rugplot along one axis of the plot."""
        vector = sub_data[var]
        n = len(vector)
        _, inv = _get_transform_functions(ax, var)
        vector = inv(vector)
        if 'hue' in self.variables:
            colors = self._hue_map(sub_data['hue'])
        else:
            colors = None
        if var == 'x':
            trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
            xy_pairs = np.column_stack([np.repeat(vector, 2), np.tile([0, height], n)])
        if var == 'y':
            trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
            xy_pairs = np.column_stack([np.tile([0, height], n), np.repeat(vector, 2)])
        line_segs = xy_pairs.reshape([n, 2, 2])
        ax.add_collection(LineCollection(line_segs, transform=trans, colors=colors, **kws))
        ax.autoscale_view(scalex=var == 'x', scaley=var == 'y')