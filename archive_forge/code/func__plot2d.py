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
def _plot2d(plotfunc):
    """Decorator for common 2d plotting logic."""
    commondoc = '\n    Parameters\n    ----------\n    darray : DataArray\n        Must be two-dimensional, unless creating faceted plots.\n    x : Hashable or None, optional\n        Coordinate for *x* axis. If ``None``, use ``darray.dims[1]``.\n    y : Hashable or None, optional\n        Coordinate for *y* axis. If ``None``, use ``darray.dims[0]``.\n    figsize : Iterable or float or None, optional\n        A tuple (width, height) of the figure in inches.\n        Mutually exclusive with ``size`` and ``ax``.\n    size : scalar, optional\n        If provided, create a new figure for the plot with the given size:\n        *height* (in inches) of each plot. See also: ``aspect``.\n    aspect : "auto", "equal", scalar or None, optional\n        Aspect ratio of plot, so that ``aspect * size`` gives the *width* in\n        inches. Only used if a ``size`` is provided.\n    ax : matplotlib axes object, optional\n        Axes on which to plot. By default, use the current axes.\n        Mutually exclusive with ``size`` and ``figsize``.\n    row : Hashable or None, optional\n        If passed, make row faceted plots on this dimension name.\n    col : Hashable or None, optional\n        If passed, make column faceted plots on this dimension name.\n    col_wrap : int, optional\n        Use together with ``col`` to wrap faceted plots.\n    xincrease : None, True, or False, optional\n        Should the values on the *x* axis be increasing from left to right?\n        If ``None``, use the default for the Matplotlib function.\n    yincrease : None, True, or False, optional\n        Should the values on the *y* axis be increasing from top to bottom?\n        If ``None``, use the default for the Matplotlib function.\n    add_colorbar : bool, optional\n        Add colorbar to axes.\n    add_labels : bool, optional\n        Use xarray metadata to label axes.\n    vmin : float or None, optional\n        Lower value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    vmax : float or None, optional\n        Upper value to anchor the colormap, otherwise it is inferred from the\n        data and other keyword arguments. When a diverging dataset is inferred,\n        setting `vmin` or `vmax` will fix the other by symmetry around\n        ``center``. Setting both values prevents use of a diverging colormap.\n        If discrete levels are provided as an explicit list, both of these\n        values are ignored.\n    cmap : matplotlib colormap name or colormap, optional\n        The mapping from data values to color space. If not provided, this\n        will be either be ``\'viridis\'`` (if the function infers a sequential\n        dataset) or ``\'RdBu_r\'`` (if the function infers a diverging dataset).\n        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:users/explain/colors/colormaps>`\n        for more information.\n\n        If *seaborn* is installed, ``cmap`` may also be a\n        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.\n        Note: if ``cmap`` is a seaborn color palette and the plot type\n        is not ``\'contour\'`` or ``\'contourf\'``, ``levels`` must also be specified.\n    center : float or False, optional\n        The value at which to center the colormap. Passing this value implies\n        use of a diverging colormap. Setting it to ``False`` prevents use of a\n        diverging colormap.\n    robust : bool, optional\n        If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is\n        computed with 2nd and 98th percentiles instead of the extreme values.\n    extend : {\'neither\', \'both\', \'min\', \'max\'}, optional\n        How to draw arrows extending the colorbar beyond its limits. If not\n        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.\n    levels : int or array-like, optional\n        Split the colormap (``cmap``) into discrete color intervals. If an integer\n        is provided, "nice" levels are chosen based on the data range: this can\n        imply that the final number of levels is not exactly the expected one.\n        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to\n        setting ``levels=np.linspace(vmin, vmax, N)``.\n    infer_intervals : bool, optional\n        Only applies to pcolormesh. If ``True``, the coordinate intervals are\n        passed to pcolormesh. If ``False``, the original coordinates are used\n        (this can be useful for certain map projections). The default is to\n        always infer intervals, unless the mesh is irregular and plotted on\n        a map projection.\n    colors : str or array-like of color-like, optional\n        A single color or a sequence of colors. If the plot type is not ``\'contour\'``\n        or ``\'contourf\'``, the ``levels`` argument is required.\n    subplot_kws : dict, optional\n        Dictionary of keyword arguments for Matplotlib subplots. Only used\n        for 2D and faceted plots.\n        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).\n    cbar_ax : matplotlib axes object, optional\n        Axes in which to draw the colorbar.\n    cbar_kwargs : dict, optional\n        Dictionary of keyword arguments to pass to the colorbar\n        (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).\n    xscale : {\'linear\', \'symlog\', \'log\', \'logit\'} or None, optional\n        Specifies scaling for the x-axes.\n    yscale : {\'linear\', \'symlog\', \'log\', \'logit\'} or None, optional\n        Specifies scaling for the y-axes.\n    xticks : ArrayLike or None, optional\n        Specify tick locations for x-axes.\n    yticks : ArrayLike or None, optional\n        Specify tick locations for y-axes.\n    xlim : tuple[float, float] or None, optional\n        Specify x-axes limits.\n    ylim : tuple[float, float] or None, optional\n        Specify y-axes limits.\n    norm : matplotlib.colors.Normalize, optional\n        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding\n        kwarg must be ``None``.\n    **kwargs : optional\n        Additional keyword arguments to wrapped Matplotlib function.\n\n    Returns\n    -------\n    artist :\n        The same type of primitive artist that the wrapped Matplotlib\n        function returns.\n    '
    plotfunc.__doc__ = f'{plotfunc.__doc__}\n{commondoc}'

    @functools.wraps(plotfunc, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def newplotfunc(darray: DataArray, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap: str | Colormap | None=None, center: float | Literal[False] | None=None, robust: bool=False, extend: ExtendOptions=None, levels: ArrayLike | None=None, infer_intervals=None, colors: str | ArrayLike | None=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> Any:
        if args:
            msg = 'Using positional arguments is deprecated for plot methods, use keyword arguments instead.'
            assert x is None
            x = args[0]
            if len(args) > 1:
                assert y is None
                y = args[1]
            if len(args) > 2:
                raise ValueError(msg)
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
        del args
        if add_colorbar is None:
            add_colorbar = True
            if plotfunc.__name__ == 'contour' or (plotfunc.__name__ == 'surface' and cmap is None):
                add_colorbar = False
        imshow_rgb = plotfunc.__name__ == 'imshow' and darray.ndim == 3 + (row is not None) + (col is not None)
        if imshow_rgb:
            add_colorbar = False
            if robust or vmax is not None or vmin is not None:
                darray = _rescale_imshow_rgb(darray.as_numpy(), vmin, vmax, robust)
                vmin, vmax, robust = (None, None, False)
        if subplot_kws is None:
            subplot_kws = dict()
        if plotfunc.__name__ == 'surface' and (not kwargs.get('_is_facetgrid', False)):
            if ax is None:
                from mpl_toolkits.mplot3d import Axes3D
                del Axes3D
                subplot_kws['projection'] = '3d'
            sharex = False
            sharey = False
        if row or col:
            allargs = locals().copy()
            del allargs['darray']
            del allargs['imshow_rgb']
            allargs.update(allargs.pop('kwargs'))
            allargs['plotfunc'] = globals()[plotfunc.__name__]
            return _easy_facetgrid(darray, kind='dataarray', **allargs)
        if darray.ndim == 0 or darray.size == 0:
            raise TypeError('No numeric data to plot.')
        if plotfunc.__name__ == 'surface' and (not kwargs.get('_is_facetgrid', False)) and (ax is not None):
            import mpl_toolkits
            if not isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
                raise ValueError('If ax is passed to surface(), it must be created with projection="3d"')
        rgb = kwargs.pop('rgb', None)
        if rgb is not None and plotfunc.__name__ != 'imshow':
            raise ValueError('The "rgb" keyword is only valid for imshow()')
        elif rgb is not None and (not imshow_rgb):
            raise ValueError('The "rgb" keyword is only valid for imshow()with a three-dimensional array (per facet)')
        xlab, ylab = _infer_xy_labels(darray=darray, x=x, y=y, imshow=imshow_rgb, rgb=rgb)
        xval = darray[xlab]
        yval = darray[ylab]
        if xval.ndim > 1 or yval.ndim > 1 or plotfunc.__name__ == 'surface':
            xval = xval.broadcast_like(darray)
            yval = yval.broadcast_like(darray)
            dims = darray.dims
        else:
            dims = (yval.dims[0], xval.dims[0])
        if imshow_rgb:
            yx_dims = (ylab, xlab)
            dims = yx_dims + tuple((d for d in darray.dims if d not in yx_dims))
        if dims != darray.dims:
            darray = darray.transpose(*dims, transpose_coords=True)
        xvalnp = xval.to_numpy()
        yvalnp = yval.to_numpy()
        zval = darray.to_masked_array(copy=False)
        xplt, xlab_extra = _resolve_intervals_2dplot(xvalnp, plotfunc.__name__)
        yplt, ylab_extra = _resolve_intervals_2dplot(yvalnp, plotfunc.__name__)
        _ensure_plottable(xplt, yplt, zval)
        cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(plotfunc, zval.data, **locals(), _is_facetgrid=kwargs.pop('_is_facetgrid', False))
        if 'contour' in plotfunc.__name__:
            kwargs['extend'] = cmap_params['extend']
            kwargs['levels'] = cmap_params['levels']
            if isinstance(colors, str):
                cmap_params['cmap'] = None
                kwargs['colors'] = colors
        if 'pcolormesh' == plotfunc.__name__:
            kwargs['infer_intervals'] = infer_intervals
            kwargs['xscale'] = xscale
            kwargs['yscale'] = yscale
        if 'imshow' == plotfunc.__name__ and isinstance(aspect, str):
            raise ValueError("plt.imshow's `aspect` kwarg is not available in xarray")
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
        primitive = plotfunc(xplt, yplt, zval, ax=ax, cmap=cmap_params['cmap'], vmin=cmap_params['vmin'], vmax=cmap_params['vmax'], norm=cmap_params['norm'], **kwargs)
        if add_labels:
            ax.set_xlabel(label_from_attrs(darray[xlab], xlab_extra))
            ax.set_ylabel(label_from_attrs(darray[ylab], ylab_extra))
            ax.set_title(darray._title_for_slice())
            if plotfunc.__name__ == 'surface':
                import mpl_toolkits
                assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D)
                ax.set_zlabel(label_from_attrs(darray))
        if add_colorbar:
            if add_labels and 'label' not in cbar_kwargs:
                cbar_kwargs['label'] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            raise ValueError("cbar_ax and cbar_kwargs can't be used with add_colorbar=False.")
        if 'origin' in kwargs:
            yincrease = None
        _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
        if np.issubdtype(xplt.dtype, np.datetime64):
            _set_concise_date(ax, 'x')
        return primitive
    del newplotfunc.__wrapped__
    return newplotfunc