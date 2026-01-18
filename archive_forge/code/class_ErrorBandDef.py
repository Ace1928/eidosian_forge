from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ErrorBandDef(CompositeMarkDef):
    """ErrorBandDef schema wrapper

    Parameters
    ----------

    type : str, :class:`ErrorBand`
        The mark type. This could a primitive mark type (one of ``"bar"``, ``"circle"``,
        ``"square"``, ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"geoshape"``,
        ``"rule"``, and ``"text"`` ) or a composite mark type ( ``"boxplot"``,
        ``"errorband"``, ``"errorbar"`` ).
    band : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    borders : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    clip : bool
        Whether a composite mark be clipped to the enclosing group’s width and height.
    color : str, dict, :class:`Color`, :class:`ExprRef`, :class:`Gradient`, :class:`HexColor`, :class:`ColorName`, :class:`LinearGradient`, :class:`RadialGradient`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:**


        * This property cannot be used in a `style config
          <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
        * The ``fill`` and ``stroke`` properties have higher precedence than ``color`` and
          will override ``color``.
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
    opacity : float
        The opacity (value between [0,1]) of the mark.
    orient : :class:`Orientation`, Literal['horizontal', 'vertical']
        Orientation of the error band. This is normally automatically determined, but can be
        specified when the orientation is ambiguous and cannot be automatically determined.
    tension : float
        The tension parameter for the interpolation type of the error band.
    """
    _schema = {'$ref': '#/definitions/ErrorBandDef'}

    def __init__(self, type: Union[str, 'SchemaBase', UndefinedType]=Undefined, band: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, borders: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, clip: Union[bool, UndefinedType]=Undefined, color: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, extent: Union['SchemaBase', Literal['ci', 'iqr', 'stderr', 'stdev'], UndefinedType]=Undefined, interpolate: Union['SchemaBase', Literal['basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open', 'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step', 'step-before', 'step-after'], UndefinedType]=Undefined, opacity: Union[float, UndefinedType]=Undefined, orient: Union['SchemaBase', Literal['horizontal', 'vertical'], UndefinedType]=Undefined, tension: Union[float, UndefinedType]=Undefined, **kwds):
        super(ErrorBandDef, self).__init__(type=type, band=band, borders=borders, clip=clip, color=color, extent=extent, interpolate=interpolate, opacity=opacity, orient=orient, tension=tension, **kwds)