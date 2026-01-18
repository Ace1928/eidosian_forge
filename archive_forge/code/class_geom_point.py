from __future__ import annotations
import typing
import numpy as np
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..scales.scale_shape import FILLED_SHAPES
from .geom import geom
@document
class geom_point(geom):
    """
    Plot points (Scatter plot)

    {usage}

    Parameters
    ----------
    {common_parameters}
    """
    DEFAULT_AES = {'alpha': 1, 'color': 'black', 'fill': None, 'shape': 'o', 'size': 1.5, 'stroke': 0.5}
    REQUIRED_AES = {'x', 'y'}
    NON_MISSING_AES = {'color', 'shape', 'size'}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        """
        Plot all groups
        """
        self.draw_group(data, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        data = coord.transform(data, panel_params)
        units = 'shape'
        for _, udata in data.groupby(units, dropna=False):
            udata.reset_index(inplace=True, drop=True)
            geom_point.draw_unit(udata, panel_params, coord, ax, **params)

    @staticmethod
    def draw_unit(data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        size = (data['size'] + data['stroke']) ** 2 * np.pi
        stroke = data['stroke'] * SIZE_FACTOR
        color = to_rgba(data['color'], data['alpha'])
        shape = data['shape'].iloc[0]
        if shape in FILLED_SHAPES:
            if all((c is None for c in data['fill'])):
                fill = color
            else:
                fill = to_rgba(data['fill'], data['alpha'])
        else:
            fill = color
            color = None
        ax.scatter(x=data['x'], y=data['y'], s=size, facecolor=fill, edgecolor=color, linewidth=stroke, marker=shape, zorder=params['zorder'], rasterized=params['raster'])

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: layer) -> DrawingArea:
        """
        Draw a point in the box

        Parameters
        ----------
        data : Series
            Data Row
        da : DrawingArea
            Canvas
        lyr : layer
            Layer

        Returns
        -------
        out : DrawingArea
        """
        from matplotlib.lines import Line2D
        if data['fill'] is None:
            data['fill'] = data['color']
        size = (data['size'] + data['stroke']) * SIZE_FACTOR
        stroke = data['stroke'] * SIZE_FACTOR
        fill = to_rgba(data['fill'], data['alpha'])
        color = to_rgba(data['color'], data['alpha'])
        key = Line2D([0.5 * da.width], [0.5 * da.height], marker=data['shape'], markersize=size, markerfacecolor=fill, markeredgecolor=color, markeredgewidth=stroke)
        da.add_artist(key)
        return da

    @staticmethod
    def legend_key_size(data: pd.Series[Any], min_size: TupleInt2, lyr: layer) -> TupleInt2:
        w, h = min_size
        pad_w, pad_h = (w * 0.5, h * 0.5)
        _size = data['size'] * SIZE_FACTOR
        _stroke = 2 * data['stroke'] * SIZE_FACTOR
        _w = _h = _size + _stroke
        if data['color'] is not None:
            w = max(w, _w + pad_w)
            h = max(h, _h + pad_h)
        return (w, h)