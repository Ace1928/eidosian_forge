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
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):
    """
    Store a set of contour lines or filled regions.

    User-callable method: `~.Axes.clabel`

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    levels : [level0, level1, ..., leveln]
        A list of floating point numbers indicating the contour levels.

    allsegs : [level0segs, level1segs, ...]
        List of all the polygon segments for all the *levels*.
        For contour lines ``len(allsegs) == len(levels)``, and for
        filled contour regions ``len(allsegs) = len(levels)-1``. The lists
        should look like ::

            level0segs = [polygon0, polygon1, ...]
            polygon0 = [[x0, y0], [x1, y1], ...]

    allkinds : ``None`` or [level0kinds, level1kinds, ...]
        Optional list of all the polygon vertex kinds (code types), as
        described and used in Path. This is used to allow multiply-
        connected paths such as holes within filled polygons.
        If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
        should look like ::

            level0kinds = [polygon0kinds, ...]
            polygon0kinds = [vertexcode0, vertexcode1, ...]

        If *allkinds* is not ``None``, usually all polygons for a
        particular contour level are grouped together so that
        ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

    **kwargs
        Keyword arguments are as described in the docstring of
        `~.Axes.contour`.

    %(contour_set_attributes)s
    """

    def __init__(self, ax, *args, levels=None, filled=False, linewidths=None, linestyles=None, hatches=(None,), alpha=None, origin=None, extent=None, cmap=None, colors=None, norm=None, vmin=None, vmax=None, extend='neither', antialiased=None, nchunk=0, locator=None, transform=None, negative_linestyles=None, clip_path=None, **kwargs):
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg *filled* is ``False`` (default) or ``True``.

        Call signature::

            ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` object to draw on.

        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the contour
            levels.

        allsegs : [level0segs, level1segs, ...]
            List of all the polygon segments for all the *levels*.
            For contour lines ``len(allsegs) == len(levels)``, and for
            filled contour regions ``len(allsegs) = len(levels)-1``. The lists
            should look like ::

                level0segs = [polygon0, polygon1, ...]
                polygon0 = [[x0, y0], [x1, y1], ...]

        allkinds : [level0kinds, level1kinds, ...], optional
            Optional list of all the polygon vertex kinds (code types), as
            described and used in Path. This is used to allow multiply-
            connected paths such as holes within filled polygons.
            If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
            should look like ::

                level0kinds = [polygon0kinds, ...]
                polygon0kinds = [vertexcode0, vertexcode1, ...]

            If *allkinds* is not ``None``, usually all polygons for a
            particular contour level are grouped together so that
            ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

        **kwargs
            Keyword arguments are as described in the docstring of
            `~.Axes.contour`.
        """
        if antialiased is None and filled:
            antialiased = False
        super().__init__(antialiaseds=antialiased, alpha=alpha, clip_path=clip_path, transform=transform)
        self.axes = ax
        self.levels = levels
        self.filled = filled
        self.hatches = hatches
        self.origin = origin
        self.extent = extent
        self.colors = colors
        self.extend = extend
        self.nchunk = nchunk
        self.locator = locator
        if isinstance(norm, mcolors.LogNorm) or isinstance(self.locator, ticker.LogLocator):
            self.logscale = True
            if norm is None:
                norm = mcolors.LogNorm()
        else:
            self.logscale = False
        _api.check_in_list([None, 'lower', 'upper', 'image'], origin=origin)
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError("If given, 'extent' must be None or (x0, x1, y0, y1)")
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image':
            self.origin = mpl.rcParams['image.origin']
        self._orig_linestyles = linestyles
        self.negative_linestyles = negative_linestyles
        if self.negative_linestyles is None:
            self.negative_linestyles = mpl.rcParams['contour.negative_linestyle']
        kwargs = self._process_args(*args, **kwargs)
        self._process_levels()
        self._extend_min = self.extend in ['min', 'both']
        self._extend_max = self.extend in ['max', 'both']
        if self.colors is not None:
            ncolors = len(self.levels)
            if self.filled:
                ncolors -= 1
            i0 = 0
            use_set_under_over = False
            total_levels = ncolors + int(self._extend_min) + int(self._extend_max)
            if len(self.colors) == total_levels and (self._extend_min or self._extend_max):
                use_set_under_over = True
                if self._extend_min:
                    i0 = 1
            cmap = mcolors.ListedColormap(self.colors[i0:None], N=ncolors)
            if use_set_under_over:
                if self._extend_min:
                    cmap.set_under(self.colors[0])
                if self._extend_max:
                    cmap.set_over(self.colors[-1])
        self.labelTexts = []
        self.labelCValues = []
        self.set_cmap(cmap)
        if norm is not None:
            self.set_norm(norm)
        with self.norm.callbacks.blocked(signal='changed'):
            if vmin is not None:
                self.norm.vmin = vmin
            if vmax is not None:
                self.norm.vmax = vmax
        self.norm._changed()
        self._process_colors()
        if self._paths is None:
            self._paths = self._make_paths_from_contour_generator()
        if self.filled:
            if linewidths is not None:
                _api.warn_external('linewidths is ignored by contourf')
            lowers, uppers = self._get_lowers_and_uppers()
            self.set(edgecolor='none', zorder=kwargs.pop('zorder', 1))
        else:
            self.set(facecolor='none', linewidths=self._process_linewidths(linewidths), linestyle=self._process_linestyles(linestyles), zorder=kwargs.pop('zorder', 2), label='_nolegend_')
        self.axes.add_collection(self, autolim=False)
        self.sticky_edges.x[:] = [self._mins[0], self._maxs[0]]
        self.sticky_edges.y[:] = [self._mins[1], self._maxs[1]]
        self.axes.update_datalim([self._mins, self._maxs])
        self.axes.autoscale_view(tight=True)
        self.changed()
        if kwargs:
            _api.warn_external('The following kwargs were not used by contour: ' + ', '.join(map(repr, kwargs)))
    allsegs = property(lambda self: [[subp.vertices for subp in p._iter_connected_components()] for p in self.get_paths()])
    allkinds = property(lambda self: [[subp.codes for subp in p._iter_connected_components()] for p in self.get_paths()])
    tcolors = _api.deprecated('3.8')(property(lambda self: [(tuple(rgba),) for rgba in self.to_rgba(self.cvalues, self.alpha)]))
    tlinewidths = _api.deprecated('3.8')(property(lambda self: [(w,) for w in self.get_linewidths()]))
    alpha = property(lambda self: self.get_alpha())
    linestyles = property(lambda self: self._orig_linestyles)

    @_api.deprecated('3.8', alternative='set_antialiased or get_antialiased', addendum='Note that get_antialiased returns an array.')
    @property
    def antialiased(self):
        return all(self.get_antialiased())

    @antialiased.setter
    def antialiased(self, aa):
        self.set_antialiased(aa)

    @_api.deprecated('3.8')
    @property
    def collections(self):
        if not hasattr(self, '_old_style_split_collections'):
            self.set_visible(False)
            fcs = self.get_facecolor()
            ecs = self.get_edgecolor()
            lws = self.get_linewidth()
            lss = self.get_linestyle()
            self._old_style_split_collections = []
            for idx, path in enumerate(self._paths):
                pc = mcoll.PathCollection([path] if len(path.vertices) else [], alpha=self.get_alpha(), antialiaseds=self._antialiaseds[idx % len(self._antialiaseds)], transform=self.get_transform(), zorder=self.get_zorder(), label='_nolegend_', facecolor=fcs[idx] if len(fcs) else 'none', edgecolor=ecs[idx] if len(ecs) else 'none', linewidths=[lws[idx % len(lws)]], linestyles=[lss[idx % len(lss)]])
                if self.filled:
                    pc.set(hatch=self.hatches[idx % len(self.hatches)])
                self._old_style_split_collections.append(pc)
            for col in self._old_style_split_collections:
                self.axes.add_collection(col)
        return self._old_style_split_collections

    def get_transform(self):
        """Return the `.Transform` instance used by this ContourSet."""
        if self._transform is None:
            self._transform = self.axes.transData
        elif not isinstance(self._transform, mtransforms.Transform) and hasattr(self._transform, '_as_mpl_transform'):
            self._transform = self._transform._as_mpl_transform(self.axes)
        return self._transform

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_contour_generator'] = None
        return state

    def legend_elements(self, variable_name='x', str_format=str):
        """
        Return a list of artists and labels suitable for passing through
        to `~.Axes.legend` which represent this ContourSet.

        The labels have the form "0 < x <= 1" stating the data ranges which
        the artists represent.

        Parameters
        ----------
        variable_name : str
            The string used inside the inequality used on the labels.
        str_format : function: float -> str
            Function used to format the numbers in the labels.

        Returns
        -------
        artists : list[`.Artist`]
            A list of the artists.
        labels : list[str]
            A list of the labels.
        """
        artists = []
        labels = []
        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            n_levels = len(self._paths)
            for idx in range(n_levels):
                artists.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=self.get_facecolor()[idx], hatch=self.hatches[idx % len(self.hatches)]))
                lower = str_format(lowers[idx])
                upper = str_format(uppers[idx])
                if idx == 0 and self.extend in ('min', 'both'):
                    labels.append(f'${variable_name} \\leq {lower}s$')
                elif idx == n_levels - 1 and self.extend in ('max', 'both'):
                    labels.append(f'${variable_name} > {upper}s$')
                else:
                    labels.append(f'${lower} < {variable_name} \\leq {upper}$')
        else:
            for idx, level in enumerate(self.levels):
                artists.append(Line2D([], [], color=self.get_edgecolor()[idx], linewidth=self.get_linewidths()[idx], linestyle=self.get_linestyles()[idx]))
                labels.append(f'${variable_name} = {str_format(level)}$')
        return (artists, labels)

    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update axes limits.
        """
        self.levels = args[0]
        allsegs = args[1]
        allkinds = args[2] if len(args) > 2 else None
        self.zmax = np.max(self.levels)
        self.zmin = np.min(self.levels)
        if allkinds is None:
            allkinds = [[None] * len(segs) for segs in allsegs]
        if self.filled:
            if len(allsegs) != len(self.levels) - 1:
                raise ValueError('must be one less number of segments as levels')
        elif len(allsegs) != len(self.levels):
            raise ValueError('must be same number of segments as levels')
        if len(allkinds) != len(allsegs):
            raise ValueError('allkinds has different length to allsegs')
        flatseglist = [s for seg in allsegs for s in seg]
        points = np.concatenate(flatseglist, axis=0)
        self._mins = points.min(axis=0)
        self._maxs = points.max(axis=0)
        self._paths = [Path.make_compound_path(*map(Path, segs, kinds)) for segs, kinds in zip(allsegs, allkinds)]
        return kwargs

    def _make_paths_from_contour_generator(self):
        """Compute ``paths`` using C extension."""
        if self._paths is not None:
            return self._paths
        paths = []
        empty_path = Path(np.empty((0, 2)))
        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            for level, level_upper in zip(lowers, uppers):
                vertices, kinds = self._contour_generator.create_filled_contour(level, level_upper)
                paths.append(Path(np.concatenate(vertices), np.concatenate(kinds)) if len(vertices) else empty_path)
        else:
            for level in self.levels:
                vertices, kinds = self._contour_generator.create_contour(level)
                paths.append(Path(np.concatenate(vertices), np.concatenate(kinds)) if len(vertices) else empty_path)
        return paths

    def _get_lowers_and_uppers(self):
        """
        Return ``(lowers, uppers)`` for filled contours.
        """
        lowers = self._levels[:-1]
        if self.zmin == lowers[0]:
            lowers = lowers.copy()
            if self.logscale:
                lowers[0] = 0.99 * self.zmin
            else:
                lowers[0] -= 1
        uppers = self._levels[1:]
        return (lowers, uppers)

    def changed(self):
        if not hasattr(self, 'cvalues'):
            self._process_colors()
        self.norm.autoscale_None(self.levels)
        self.set_array(self.cvalues)
        self.update_scalarmappable()
        alphas = np.broadcast_to(self.get_alpha(), len(self.cvalues))
        for label, cv, alpha in zip(self.labelTexts, self.labelCValues, alphas):
            label.set_alpha(alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        super().changed()

    def _autolev(self, N):
        """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)
        lev = self.locator.tick_values(self.zmin, self.zmax)
        try:
            if self.locator._symmetric:
                return lev
        except AttributeError:
            pass
        under = np.nonzero(lev < self.zmin)[0]
        i0 = under[-1] if len(under) else 0
        over = np.nonzero(lev > self.zmax)[0]
        i1 = over[0] + 1 if len(over) else len(lev)
        if self.extend in ('min', 'both'):
            i0 += 1
        if self.extend in ('max', 'both'):
            i1 -= 1
        if i1 - i0 < 3:
            i0, i1 = (0, len(lev))
        return lev[i0:i1]

    def _process_contour_level_args(self, args, z_dtype):
        """
        Determine the contour levels and store in self.levels.
        """
        if self.levels is None:
            if args:
                levels_arg = args[0]
            elif np.issubdtype(z_dtype, bool):
                if self.filled:
                    levels_arg = [0, 0.5, 1]
                else:
                    levels_arg = [0.5]
            else:
                levels_arg = 7
        else:
            levels_arg = self.levels
        if isinstance(levels_arg, Integral):
            self.levels = self._autolev(levels_arg)
        else:
            self.levels = np.asarray(levels_arg, np.float64)
        if self.filled and len(self.levels) < 2:
            raise ValueError('Filled contours require at least 2 levels.')
        if len(self.levels) > 1 and np.min(np.diff(self.levels)) <= 0.0:
            raise ValueError('Contour levels must be increasing')

    def _process_levels(self):
        """
        Assign values to :attr:`layers` based on :attr:`levels`,
        adding extended layers as needed if contours are filled.

        For line contours, layers simply coincide with levels;
        a line is a thin layer.  No extended levels are needed
        with line contours.
        """
        self._levels = list(self.levels)
        if self.logscale:
            lower, upper = (1e-250, 1e+250)
        else:
            lower, upper = (-1e+250, 1e+250)
        if self.extend in ('both', 'min'):
            self._levels.insert(0, lower)
        if self.extend in ('both', 'max'):
            self._levels.append(upper)
        self._levels = np.asarray(self._levels)
        if not self.filled:
            self.layers = self.levels
            return
        if self.logscale:
            self.layers = np.sqrt(self._levels[:-1]) * np.sqrt(self._levels[1:])
        else:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the colormapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            i0, i1 = (0, len(self.levels))
            if self.filled:
                i1 -= 1
                if self.extend in ('both', 'min'):
                    i0 -= 1
                if self.extend in ('both', 'max'):
                    i1 += 1
            self.cvalues = list(range(i0, i1))
            self.set_norm(mcolors.NoNorm())
        else:
            self.cvalues = self.layers
        self.norm.autoscale_None(self.levels)
        self.set_array(self.cvalues)
        self.update_scalarmappable()
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False

    def _process_linewidths(self, linewidths):
        Nlev = len(self.levels)
        if linewidths is None:
            default_linewidth = mpl.rcParams['contour.linewidth']
            if default_linewidth is None:
                default_linewidth = mpl.rcParams['lines.linewidth']
            return [default_linewidth] * Nlev
        elif not np.iterable(linewidths):
            return [linewidths] * Nlev
        else:
            linewidths = list(linewidths)
            return (linewidths * math.ceil(Nlev / len(linewidths)))[:Nlev]

    def _process_linestyles(self, linestyles):
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            if self.monochrome:
                eps = -(self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = self.negative_linestyles
        elif isinstance(linestyles, str):
            tlinestyles = [linestyles] * Nlev
        elif np.iterable(linestyles):
            tlinestyles = list(linestyles)
            if len(tlinestyles) < Nlev:
                nreps = int(np.ceil(Nlev / len(linestyles)))
                tlinestyles = tlinestyles * nreps
            if len(tlinestyles) > Nlev:
                tlinestyles = tlinestyles[:Nlev]
        else:
            raise ValueError('Unrecognized type for linestyles kwarg')
        return tlinestyles

    def _find_nearest_contour(self, xy, indices=None):
        """
        Find the point in the unfilled contour plot that is closest (in screen
        space) to point *xy*.

        Parameters
        ----------
        xy : tuple[float, float]
            The reference point (in screen space).
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all levels
            are considered.

        Returns
        -------
        idx_level_min : int
            The index of the contour level closest to *xy*.
        idx_vtx_min : int
            The index of the `.Path` segment closest to *xy* (at that level).
        proj : (float, float)
            The point in the contour plot closest to *xy*.
        """
        if self.filled:
            raise ValueError('Method does not support filled contours')
        if indices is None:
            indices = range(len(self._paths))
        d2min = np.inf
        idx_level_min = idx_vtx_min = proj_min = None
        for idx_level in indices:
            path = self._paths[idx_level]
            idx_vtx_start = 0
            for subpath in path._iter_connected_components():
                if not len(subpath.vertices):
                    continue
                lc = self.get_transform().transform(subpath.vertices)
                d2, proj, leg = _find_closest_point_on_path(lc, xy)
                if d2 < d2min:
                    d2min = d2
                    idx_level_min = idx_level
                    idx_vtx_min = leg[1] + idx_vtx_start
                    proj_min = proj
                idx_vtx_start += len(subpath)
        return (idx_level_min, idx_vtx_min, proj_min)

    def find_nearest_contour(self, x, y, indices=None, pixel=True):
        """
        Find the point in the contour plot that is closest to ``(x, y)``.

        This method does not support filled contours.

        Parameters
        ----------
        x, y : float
            The reference point.
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all
            levels are considered.
        pixel : bool, default: True
            If *True*, measure distance in pixel (screen) space, which is
            useful for manual contour labeling; else, measure distance in axes
            space.

        Returns
        -------
        path : int
            The index of the path that is closest to ``(x, y)``.  Each path corresponds
            to one contour level.
        subpath : int
            The index within that closest path of the subpath that is closest to
            ``(x, y)``.  Each subpath corresponds to one unbroken contour line.
        index : int
            The index of the vertices within that subpath that are closest to
            ``(x, y)``.
        xmin, ymin : float
            The point in the contour plot that is closest to ``(x, y)``.
        d2 : float
            The squared distance from ``(xmin, ymin)`` to ``(x, y)``.
        """
        segment = index = d2 = None
        with ExitStack() as stack:
            if not pixel:
                stack.enter_context(self._cm_set(transform=mtransforms.IdentityTransform()))
            i_level, i_vtx, (xmin, ymin) = self._find_nearest_contour((x, y), indices)
        if i_level is not None:
            cc_cumlens = np.cumsum([*map(len, self._paths[i_level]._iter_connected_components())])
            segment = cc_cumlens.searchsorted(i_vtx, 'right')
            index = i_vtx if segment == 0 else i_vtx - cc_cumlens[segment - 1]
            d2 = (xmin - x) ** 2 + (ymin - y) ** 2
        return (i_level, segment, index, xmin, ymin, d2)

    def draw(self, renderer):
        paths = self._paths
        n_paths = len(paths)
        if not self.filled or all((hatch is None for hatch in self.hatches)):
            super().draw(renderer)
            return
        for idx in range(n_paths):
            with cbook._setattr_cm(self, _paths=[paths[idx]]), self._cm_set(hatch=self.hatches[idx % len(self.hatches)], array=[self.get_array()[idx]], linewidths=[self.get_linewidths()[idx % len(self.get_linewidths())]], linestyles=[self.get_linestyles()[idx % len(self.get_linestyles())]]):
                super().draw(renderer)