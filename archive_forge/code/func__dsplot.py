from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from xarray.core.alignment import broadcast
from xarray.plot import dataarray_plot
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _dsplot(plotfunc):
    commondoc = '\n    Parameters\n    ----------\n\n    ds : Dataset\n    x : Hashable or None, optional\n        Variable name for x-axis.\n    y : Hashable or None, optional\n        Variable name for y-axis.\n    u : Hashable or None, optional\n        Variable name for the *u* velocity (in *x* direction).\n        quiver/streamplot plots only.\n    v : Hashable or None, optional\n        Variable name for the *v* velocity (in *y* direction).\n        quiver/streamplot plots only.\n    hue: Hashable or None, optional\n        Variable by which to color scatter points or arrows.\n    hue_style: {\'continuous\', \'discrete\'} or None, optional\n        How to use the ``hue`` variable:\n\n        - ``\'continuous\'`` -- continuous color scale\n          (default for numeric ``hue`` variables)\n        - ``\'discrete\'`` -- a color for each unique value, using the default color cycle\n          (default for non-numeric ``hue`` variables)\n\n    row : Hashable or None, optional\n        If passed, make row faceted plots on this dimension name.\n    col : Hashable or None, optional\n        If passed, make column faceted plots on this dimension name.\n    col_wrap : int, optional\n        Use together with ``col`` to wrap faceted plots.\n    ax : matplotlib axes object or None, optional\n        If ``None``, use the current axes. Not applicable when using facets.\n    figsize : Iterable[float] or None, optional\n        A tuple (width, height) of the figure in inches.\n        Mutually exclusive with ``size`` and ``ax``.\n    size : scalar, optional\n        If provided, create a new figure for the plot with the given size.\n        Height (in inches) of each plot. See also: ``aspect``.\n    aspect : "auto", "equal", scalar or None, optional\n        Aspect ratio of plot, so that ``aspect * size`` gives the width in\n        inches. Only used if a ``size`` is provided.\n    sharex : bool or None, optional\n        If True all subplots share the same x-axis.\n    sharey : bool or None, optional\n        If True all subplots share the same y-axis.\n    add_guide: bool or None, optional\n        Add a guide that depends on ``hue_style``:\n\n        - ``\'continuous\'`` -- build a colorbar\n        - ``\'discrete\'`` -- build a legend\n\n    subplot_kws : dict or None, optional\n        Dictionary of keyword arguments for Matplotlib subplots\n        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).\n        Only applies to FacetGrid plotting.\n    cbar_kwargs : dict, optional\n        Dictionary of keyword arguments to pass to the colorbar\n        (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).\n    cbar_ax : matplotlib axes object, optional\n        Axes in which to draw the colorbar.\n    cmap : matplotlib colormap name or colormap, optional\n        The mapping from data values to color space. Either a\n        Matplotlib colormap name or object. If not provided, this will\n        be either ``\'viridis\'`` (if the function infers a sequential\n        dataset) or ``\'RdBu_r\'`` (if the function infers a diverging\n        dataset).\n        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:users/explain/colors/colormaps>`\n        for more information.\n\n        If *seaborn* is installed, ``cmap`` may also be a\n        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.\n        Note: if ``cmap`` is a seaborn color palette,\n        ``levels`` must also be specified.\n    vmin : float or None, optional\n        Lower value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    vmax : float or None, optional\n        Upper value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    norm : matplotlib.colors.Normalize, optional\n        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding\n        kwarg must be ``None``.\n    infer_intervals: bool | None\n        If True the intervals are inferred.\n    center : float, optional\n        The value at which to center the colormap. Passing this value implies\n        use of a diverging colormap. Setting it to ``False`` prevents use of a\n        diverging colormap.\n    robust : bool, optional\n        If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is\n        computed with 2nd and 98th percentiles instead of the extreme values.\n    colors : str or array-like of color-like, optional\n        A single color or a list of colors. The ``levels`` argument\n        is required.\n    extend : {\'neither\', \'both\', \'min\', \'max\'}, optional\n        How to draw arrows extending the colorbar beyond its limits. If not\n        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.\n    levels : int or array-like, optional\n        Split the colormap (``cmap``) into discrete color intervals. If an integer\n        is provided, "nice" levels are chosen based on the data range: this can\n        imply that the final number of levels is not exactly the expected one.\n        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to\n        setting ``levels=np.linspace(vmin, vmax, N)``.\n    **kwargs : optional\n        Additional keyword arguments to wrapped Matplotlib function.\n    '
    plotfunc.__doc__ = f'{plotfunc.__doc__}\n{commondoc}'

    @functools.wraps(plotfunc, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def newplotfunc(ds: Dataset, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, sharex: bool=True, sharey: bool=True, add_guide: bool | None=None, subplot_kws: dict[str, Any] | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cmap: str | Colormap | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals: bool | None=None, center: float | None=None, robust: bool | None=None, colors: str | ArrayLike | None=None, extend: ExtendOptions=None, levels: ArrayLike | None=None, **kwargs: Any) -> Any:
        if args:
            msg = 'Using positional arguments is deprecated for plot methods, use keyword arguments instead.'
            assert x is None
            x = args[0]
            if len(args) > 1:
                assert y is None
                y = args[1]
            if len(args) > 2:
                assert u is None
                u = args[2]
            if len(args) > 3:
                assert v is None
                v = args[3]
            if len(args) > 4:
                assert hue is None
                hue = args[4]
            if len(args) > 5:
                raise ValueError(msg)
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
        del args
        _is_facetgrid = kwargs.pop('_is_facetgrid', False)
        if _is_facetgrid:
            meta_data = kwargs.pop('meta_data')
        else:
            meta_data = _infer_meta_data(ds, x, y, hue, hue_style, add_guide, funcname=plotfunc.__name__)
        hue_style = meta_data['hue_style']
        if col or row:
            allargs = locals().copy()
            allargs['plotfunc'] = globals()[plotfunc.__name__]
            allargs['data'] = ds
            for arg in ['meta_data', 'kwargs', 'ds']:
                del allargs[arg]
            return _easy_facetgrid(kind='dataset', **allargs, **kwargs)
        figsize = kwargs.pop('figsize', None)
        ax = get_axis(figsize, size, aspect, ax)
        if hue_style == 'continuous' and hue is not None:
            if _is_facetgrid:
                cbar_kwargs = meta_data['cbar_kwargs']
                cmap_params = meta_data['cmap_params']
            else:
                cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(plotfunc, ds[hue].values, **locals())
            cmap_params_subset = {vv: cmap_params[vv] for vv in ['vmin', 'vmax', 'norm', 'cmap']}
        else:
            cmap_params_subset = {}
        if (u is not None or v is not None) and plotfunc.__name__ not in ('quiver', 'streamplot'):
            raise ValueError('u, v are only allowed for quiver or streamplot plots.')
        primitive = plotfunc(ds=ds, x=x, y=y, ax=ax, u=u, v=v, hue=hue, hue_style=hue_style, cmap_params=cmap_params_subset, **kwargs)
        if _is_facetgrid:
            return primitive
        if meta_data.get('xlabel', None):
            ax.set_xlabel(meta_data.get('xlabel'))
        if meta_data.get('ylabel', None):
            ax.set_ylabel(meta_data.get('ylabel'))
        if meta_data['add_legend']:
            ax.legend(handles=primitive, title=meta_data.get('hue_label', None))
        if meta_data['add_colorbar']:
            cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
            if 'label' not in cbar_kwargs:
                cbar_kwargs['label'] = meta_data.get('hue_label', None)
            _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        if meta_data['add_quiverkey']:
            magnitude = _get_nice_quiver_magnitude(ds[u], ds[v])
            units = ds[u].attrs.get('units', '')
            ax.quiverkey(primitive, X=0.85, Y=0.9, U=magnitude, label=f'{magnitude}\n{units}', labelpos='E', coordinates='figure')
        if plotfunc.__name__ in ('quiver', 'streamplot'):
            title = ds[u]._title_for_slice()
        else:
            title = ds[x]._title_for_slice()
        ax.set_title(title)
        return primitive
    del newplotfunc.__wrapped__
    return newplotfunc