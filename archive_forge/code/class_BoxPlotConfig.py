from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BoxPlotConfig(VegaLiteSchema):
    """BoxPlotConfig schema wrapper

    Parameters
    ----------

    box : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    extent : str, float
        The extent of the whiskers. Available options include:


        * ``"min-max"`` : min and max are the lower and upper whiskers respectively.
        * A number representing multiple of the interquartile range. This number will be
          multiplied by the IQR to determine whisker boundary, which spans from the smallest
          data to the largest data within the range *[Q1 - k * IQR, Q3 + k * IQR]* where
          *Q1* and *Q3* are the first and third quartiles while *IQR* is the interquartile
          range ( *Q3-Q1* ).

        **Default value:** ``1.5``.
    median : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    outliers : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    rule : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    size : float
        Size of the box and median tick of a box plot
    ticks : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    """
    _schema = {'$ref': '#/definitions/BoxPlotConfig'}

    def __init__(self, box: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, extent: Union[str, float, UndefinedType]=Undefined, median: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, outliers: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, rule: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, size: Union[float, UndefinedType]=Undefined, ticks: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(BoxPlotConfig, self).__init__(box=box, extent=extent, median=median, outliers=outliers, rule=rule, size=size, ticks=ticks, **kwds)