from __future__ import annotations
import typing
from .._utils import SIZE_FACTOR, to_rgba
from ..coords import coord_flip
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon
@staticmethod
def _draw_outline(data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
    outline_type = params['outline_type']
    if outline_type == 'full':
        return
    x, y = ('x', 'y')
    if isinstance(coord, coord_flip):
        x, y = (y, x)
        data[x], data[y] = (data[y], data[x])
    if outline_type in ('lower', 'both'):
        geom_path.draw_group(data.eval(f'y = {y}min'), panel_params, coord, ax, **params)
    if outline_type in ('upper', 'both'):
        geom_path.draw_group(data.eval(f'y = {y}max'), panel_params, coord, ax, **params)