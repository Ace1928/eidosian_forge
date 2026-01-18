from __future__ import annotations
import typing
from .position import position
class position_nudge(position):
    """
    Nudge points

    Useful to nudge labels away from the points
    being labels.

    Parameters
    ----------
    x :
        Horizontal nudge
    y :
        Vertical nudge
    """

    def __init__(self, x: float=0, y: float=0):
        self.params = {'x': x, 'y': y}

    @classmethod
    def compute_layer(cls, data, params, layout):
        trans_x = None
        trans_y = None
        if params['x']:

            def trans_x(x: FloatArrayLike) -> FloatArray:
                return x + params['x']
        if params['y']:

            def trans_y(y: FloatArrayLike) -> FloatArray:
                return y + params['y']
        return cls.transform_position(data, trans_x, trans_y)