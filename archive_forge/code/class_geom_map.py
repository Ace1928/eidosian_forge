from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_point import geom_point
from .geom_polygon import geom_polygon
@document
class geom_map(geom):
    """
    Draw map feature

    The map feature are drawn without any special projections.

    {usage}

    Parameters
    ----------
    {common_parameters}

    Notes
    -----
    This geom is best suited for plotting a shapefile read into
    geopandas dataframe. The dataframe should have a `geometry`
    column.
    """
    DEFAULT_AES = {'alpha': 1, 'color': '#111111', 'fill': '#333333', 'linetype': 'solid', 'shape': 'o', 'size': 0.5, 'stroke': 0.5}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}
    REQUIRED_AES = {'geometry'}

    def __init__(self, mapping: aes | None=None, data: DataLike | None=None, **kwargs: Any):
        geom.__init__(self, mapping, data, **kwargs)
        if 'geometry' not in self.mapping:
            self.mapping['geometry'] = 'geometry'

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if not len(data):
            return data
        bool_idx = np.array([g is not None for g in data['geometry']])
        if not np.all(bool_idx):
            data = data.loc[bool_idx]
        try:
            bounds = data['geometry'].bounds
        except AttributeError:
            bounds = pd.DataFrame(np.array([x.bounds for x in data['geometry']]), columns=['xmin', 'ymin', 'xmax', 'ymax'], index=data.index)
        else:
            bounds.rename(columns={'minx': 'xmin', 'maxx': 'xmax', 'miny': 'ymin', 'maxy': 'ymax'}, inplace=True)
        data = pd.concat([data, bounds], axis=1)
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        if not len(data):
            return
        data.loc[data['color'].isna(), 'color'] = 'none'
        data.loc[data['fill'].isna(), 'fill'] = 'none'
        data['fill'] = to_rgba(data['fill'], data['alpha'])
        geom_type = data.geometry.iloc[0].geom_type
        if geom_type in ('Polygon', 'MultiPolygon'):
            from matplotlib.collections import PatchCollection
            data['size'] *= SIZE_FACTOR
            patches = [PolygonPatch(g) for g in data['geometry']]
            coll = PatchCollection(patches, edgecolor=data['color'], facecolor=data['fill'], linestyle=data['linetype'], linewidth=data['size'], zorder=params['zorder'], rasterized=params['raster'])
            ax.add_collection(coll)
        elif geom_type == 'Point':
            arr = np.array([list(g.coords)[0] for g in data['geometry']])
            data['x'] = arr[:, 0]
            data['y'] = arr[:, 1]
            for _, gdata in data.groupby('group'):
                gdata.reset_index(inplace=True, drop=True)
                gdata.is_copy = None
                geom_point.draw_group(gdata, panel_params, coord, ax, **params)
        elif geom_type == 'MultiPoint':
            data['points'] = [[p.coords[0] for p in mp.geoms] for mp in data['geometry']]
            data = data.explode('points', ignore_index=True)
            data['x'] = [p[0] for p in data['points']]
            data['y'] = [p[1] for p in data['points']]
            geom_point.draw_group(data, panel_params, coord, ax, **params)
        elif geom_type in ('LineString', 'MultiLineString'):
            from matplotlib.collections import LineCollection
            data['size'] *= SIZE_FACTOR
            data['color'] = to_rgba(data['color'], data['alpha'])
            segments = []
            for g in data['geometry']:
                if g.geom_type == 'LineString':
                    segments.append(g.coords)
                else:
                    segments.extend((_g.coords for _g in g.geoms))
            coll = LineCollection(segments, edgecolor=data['color'], linewidth=data['size'], linestyle=data['linetype'], zorder=params['zorder'], rasterized=params['raster'])
            ax.add_collection(coll)
        else:
            raise TypeError(f"Could not plot geometry of type '{geom_type}'")

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: layer) -> DrawingArea:
        """
        Draw a rectangle in the box

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
        data['size'] = data['stroke']
        del data['stroke']
        return geom_polygon.draw_legend(data, da, lyr)