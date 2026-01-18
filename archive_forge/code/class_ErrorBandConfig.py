from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ErrorBandConfig(VegaLiteSchema):
    """ErrorBandConfig schema wrapper

    Parameters
    ----------

    band : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    borders : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    extent : :class:`ErrorBarExtent`, Literal['ci', 'iqr', 'stderr', 'stdev']
        The extent of the band. Available options include:


        * ``"ci"`` : Extend the band to the confidence interval of the mean.
        * ``"stderr"`` : The size of band are set to the value of standard error, extending
          from the mean.
        * ``"stdev"`` : The size of band are set to the value of standard deviation,
          extending from the mean.
        * ``"iqr"`` : Extend the band to the q1 and q3.

        **Default value:** ``"stderr"``.
    interpolate : :class:`Interpolate`, Literal['basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open', 'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step', 'step-before', 'step-after']
        The line interpolation method for the error band. One of the following:


        * ``"linear"`` : piecewise linear segments, as in a polyline.
        * ``"linear-closed"`` : close the linear segments to form a polygon.
        * ``"step"`` : a piecewise constant function (a step function) consisting of
          alternating horizontal and vertical lines. The y-value changes at the midpoint of
          each pair of adjacent x-values.
        * ``"step-before"`` : a piecewise constant function (a step function) consisting of
          alternating horizontal and vertical lines. The y-value changes before the x-value.
        * ``"step-after"`` : a piecewise constant function (a step function) consisting of
          alternating horizontal and vertical lines. The y-value changes after the x-value.
        * ``"basis"`` : a B-spline, with control point duplication on the ends.
        * ``"basis-open"`` : an open B-spline; may not intersect the start or end.
        * ``"basis-closed"`` : a closed B-spline, as in a loop.
        * ``"cardinal"`` : a Cardinal spline, with control point duplication on the ends.
        * ``"cardinal-open"`` : an open Cardinal spline; may not intersect the start or end,
          but will intersect other control points.
        * ``"cardinal-closed"`` : a closed Cardinal spline, as in a loop.
        * ``"bundle"`` : equivalent to basis, except the tension parameter is used to
          straighten the spline.
        * ``"monotone"`` : cubic interpolation that preserves monotonicity in y.
    tension : float
        The tension parameter for the interpolation type of the error band.
    """
    _schema = {'$ref': '#/definitions/ErrorBandConfig'}

    def __init__(self, band: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, borders: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, extent: Union['SchemaBase', Literal['ci', 'iqr', 'stderr', 'stdev'], UndefinedType]=Undefined, interpolate: Union['SchemaBase', Literal['basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open', 'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step', 'step-before', 'step-after'], UndefinedType]=Undefined, tension: Union[float, UndefinedType]=Undefined, **kwds):
        super(ErrorBandConfig, self).__init__(band=band, borders=borders, extent=extent, interpolate=interpolate, tension=tension, **kwds)