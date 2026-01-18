from __future__ import annotations
import functools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, overload
import numpy as np
import pandas as pd
from xarray.core.alignment import broadcast
from xarray.core.concat import concat
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _plot1d(plotfunc):
    """Decorator for common 1d plotting logic."""
    commondoc = '\n    Parameters\n    ----------\n    darray : DataArray\n        Must be 2 dimensional, unless creating faceted plots.\n    x : Hashable or None, optional\n        Coordinate for x axis. If None use darray.dims[1].\n    y : Hashable or None, optional\n        Coordinate for y axis. If None use darray.dims[0].\n    z : Hashable or None, optional\n        If specified plot 3D and use this coordinate for *z* axis.\n    hue : Hashable or None, optional\n        Dimension or coordinate for which you want multiple lines plotted.\n    markersize: Hashable or None, optional\n        scatter only. Variable by which to vary size of scattered points.\n    linewidth: Hashable or None, optional\n        Variable by which to vary linewidth.\n    row : Hashable, optional\n        If passed, make row faceted plots on this dimension name.\n    col : Hashable, optional\n        If passed, make column faceted plots on this dimension name.\n    col_wrap : int, optional\n        Use together with ``col`` to wrap faceted plots\n    ax : matplotlib axes object, optional\n        If None, uses the current axis. Not applicable when using facets.\n    figsize : Iterable[float] or None, optional\n        A tuple (width, height) of the figure in inches.\n        Mutually exclusive with ``size`` and ``ax``.\n    size : scalar, optional\n        If provided, create a new figure for the plot with the given size.\n        Height (in inches) of each plot. See also: ``aspect``.\n    aspect : "auto", "equal", scalar or None, optional\n        Aspect ratio of plot, so that ``aspect * size`` gives the width in\n        inches. Only used if a ``size`` is provided.\n    xincrease : bool or None, default: True\n        Should the values on the x axes be increasing from left to right?\n        if None, use the default for the matplotlib function.\n    yincrease : bool or None, default: True\n        Should the values on the y axes be increasing from top to bottom?\n        if None, use the default for the matplotlib function.\n    add_legend : bool or None, optional\n        If True use xarray metadata to add a legend.\n    add_colorbar : bool or None, optional\n        If True add a colorbar.\n    add_labels : bool or None, optional\n        If True use xarray metadata to label axes\n    add_title : bool or None, optional\n        If True use xarray metadata to add a title\n    subplot_kws : dict, optional\n        Dictionary of keyword arguments for matplotlib subplots. Only applies\n        to FacetGrid plotting.\n    xscale : {\'linear\', \'symlog\', \'log\', \'logit\'} or None, optional\n        Specifies scaling for the x-axes.\n    yscale : {\'linear\', \'symlog\', \'log\', \'logit\'} or None, optional\n        Specifies scaling for the y-axes.\n    xticks : ArrayLike or None, optional\n        Specify tick locations for x-axes.\n    yticks : ArrayLike or None, optional\n        Specify tick locations for y-axes.\n    xlim : tuple[float, float] or None, optional\n        Specify x-axes limits.\n    ylim : tuple[float, float] or None, optional\n        Specify y-axes limits.\n    cmap : matplotlib colormap name or colormap, optional\n        The mapping from data values to color space. Either a\n        Matplotlib colormap name or object. If not provided, this will\n        be either ``\'viridis\'`` (if the function infers a sequential\n        dataset) or ``\'RdBu_r\'`` (if the function infers a diverging\n        dataset).\n        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:users/explain/colors/colormaps>`\n        for more information.\n\n        If *seaborn* is installed, ``cmap`` may also be a\n        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.\n        Note: if ``cmap`` is a seaborn color palette,\n        ``levels`` must also be specified.\n    vmin : float or None, optional\n        Lower value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    vmax : float or None, optional\n        Upper value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    norm : matplotlib.colors.Normalize, optional\n        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding\n        kwarg must be ``None``.\n    extend : {\'neither\', \'both\', \'min\', \'max\'}, optional\n        How to draw arrows extending the colorbar beyond its limits. If not\n        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.\n    levels : int or array-like, optional\n        Split the colormap (``cmap``) into discrete color intervals. If an integer\n        is provided, "nice" levels are chosen based on the data range: this can\n        imply that the final number of levels is not exactly the expected one.\n        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to\n        setting ``levels=np.linspace(vmin, vmax, N)``.\n    **kwargs : optional\n        Additional arguments to wrapped matplotlib function\n\n    Returns\n    -------\n    artist :\n        The same type of primitive artist that the wrapped matplotlib\n        function returns\n    '
    plotfunc.__doc__ = f'{plotfunc.__doc__}\n{commondoc}'

    @functools.wraps(plotfunc, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def newplotfunc(darray: DataArray, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap: str | Colormap | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend: ExtendOptions=None, levels: ArrayLike | None=None, **kwargs) -> Any:
        import matplotlib.pyplot as plt
        if subplot_kws is None:
            subplot_kws = dict()
        if row or col:
            if z is not None:
                subplot_kws.update(projection='3d')
            allargs = locals().copy()
            allargs.update(allargs.pop('kwargs'))
            allargs.pop('darray')
            allargs.pop('plt')
            allargs['plotfunc'] = globals()[plotfunc.__name__]
            return _easy_facetgrid(darray, kind='plot1d', **allargs)
        if darray.ndim == 0 or darray.size == 0:
            raise TypeError('No numeric data to plot.')
        if args == ():
            args = kwargs.pop('args', ())
        if args:
            assert 'args' not in kwargs
            msg = 'Using positional arguments is deprecated for plot methods, use keyword arguments instead.'
            assert x is None
            x = args[0]
            if len(args) > 1:
                assert y is None
                y = args[1]
            if len(args) > 2:
                assert z is None
                z = args[2]
            if len(args) > 3:
                assert hue is None
                hue = args[3]
            if len(args) > 4:
                raise ValueError(msg)
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
        del args
        if hue_style is not None:
            warnings.warn('hue_style is no longer used for plot1d plots and the argument will eventually be removed. Convert numbers to string for a discrete hue and use add_legend or add_colorbar to control which guide to display.', DeprecationWarning, stacklevel=2)
        _is_facetgrid = kwargs.pop('_is_facetgrid', False)
        if plotfunc.__name__ == 'scatter':
            size_ = kwargs.pop('_size', markersize)
            size_r = _MARKERSIZE_RANGE
            darray = darray.load()
            darray = darray.where(darray.notnull(), drop=True)
        else:
            size_ = kwargs.pop('_size', linewidth)
            size_r = _LINEWIDTH_RANGE
        coords_to_plot: MutableMapping[str, Hashable | None] = dict(x=x, z=z, hue=hue, size=size_)
        if not _is_facetgrid:
            coords_to_plot = _guess_coords_to_plot(darray, coords_to_plot, kwargs)
        plts = _prepare_plot1d_data(darray, coords_to_plot, plotfunc.__name__)
        xplt = plts.pop('x', None)
        yplt = plts.pop('y', None)
        zplt = plts.pop('z', None)
        kwargs.update(zplt=zplt)
        hueplt = plts.pop('hue', None)
        sizeplt = plts.pop('size', None)
        hueplt_norm = _Normalize(data=hueplt)
        kwargs.update(hueplt=hueplt_norm.values)
        sizeplt_norm = _Normalize(data=sizeplt, width=size_r, _is_facetgrid=_is_facetgrid)
        kwargs.update(sizeplt=sizeplt_norm.values)
        cmap_params_subset = kwargs.pop('cmap_params_subset', {})
        cbar_kwargs = kwargs.pop('cbar_kwargs', {})
        if hueplt_norm.data is not None:
            if not hueplt_norm.data_is_numeric:
                cbar_kwargs.update(format=hueplt_norm.format, ticks=hueplt_norm.ticks)
                levels = kwargs.get('levels', hueplt_norm.levels)
            cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(plotfunc, cast('DataArray', hueplt_norm.values).data, **locals())
            if not cmap_params_subset:
                ckw = {vv: cmap_params[vv] for vv in ('vmin', 'vmax', 'norm', 'cmap')}
                cmap_params_subset.update(**ckw)
        with plt.rc_context(_styles):
            if z is not None:
                import mpl_toolkits
                if ax is None:
                    subplot_kws.update(projection='3d')
                ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
                assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D)
                ax.view_init(azim=30, elev=30, vertical_axis='y')
            else:
                ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
            primitive = plotfunc(xplt, yplt, ax=ax, add_labels=add_labels, **cmap_params_subset, **kwargs)
        if np.any(np.asarray(add_labels)) and add_title:
            ax.set_title(darray._title_for_slice())
        add_colorbar_, add_legend_ = _determine_guide(hueplt_norm, sizeplt_norm, add_colorbar, add_legend, plotfunc_name=plotfunc.__name__)
        if add_colorbar_:
            if 'label' not in cbar_kwargs:
                cbar_kwargs['label'] = label_from_attrs(hueplt_norm.data)
            _add_colorbar(primitive, ax, kwargs.get('cbar_ax', None), cbar_kwargs, cmap_params)
        if add_legend_:
            if plotfunc.__name__ in ['scatter', 'line']:
                _add_legend(hueplt_norm if add_legend or not add_colorbar_ else _Normalize(None), sizeplt_norm, primitive, legend_ax=ax, plotfunc=plotfunc.__name__)
            else:
                hueplt_norm_values: list[np.ndarray | None]
                if hueplt_norm.data is not None:
                    hueplt_norm_values = list(hueplt_norm.data.to_numpy())
                else:
                    hueplt_norm_values = [hueplt_norm.data]
                if plotfunc.__name__ == 'hist':
                    ax.legend(handles=primitive[-1], labels=hueplt_norm_values, title=label_from_attrs(hueplt_norm.data))
                else:
                    ax.legend(handles=primitive, labels=hueplt_norm_values, title=label_from_attrs(hueplt_norm.data))
        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
        return primitive
    del newplotfunc.__wrapped__
    return newplotfunc