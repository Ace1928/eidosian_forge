from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ErrorBarConfig(VegaLiteSchema):
    """ErrorBarConfig schema wrapper

    Parameters
    ----------

    extent : :class:`ErrorBarExtent`, Literal['ci', 'iqr', 'stderr', 'stdev']
        The extent of the rule. Available options include:


        * ``"ci"`` : Extend the rule to the confidence interval of the mean.
        * ``"stderr"`` : The size of rule are set to the value of standard error, extending
          from the mean.
        * ``"stdev"`` : The size of rule are set to the value of standard deviation,
          extending from the mean.
        * ``"iqr"`` : Extend the rule to the q1 and q3.

        **Default value:** ``"stderr"``.
    rule : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    size : float
        Size of the ticks of an error bar
    thickness : float
        Thickness of the ticks and the bar of an error bar
    ticks : bool, dict, :class:`BarConfig`, :class:`AreaConfig`, :class:`LineConfig`, :class:`MarkConfig`, :class:`RectConfig`, :class:`TickConfig`, :class:`AnyMarkConfig`

    """
    _schema = {'$ref': '#/definitions/ErrorBarConfig'}

    def __init__(self, extent: Union['SchemaBase', Literal['ci', 'iqr', 'stderr', 'stdev'], UndefinedType]=Undefined, rule: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, size: Union[float, UndefinedType]=Undefined, thickness: Union[float, UndefinedType]=Undefined, ticks: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(ErrorBarConfig, self).__init__(extent=extent, rule=rule, size=size, thickness=thickness, ticks=ticks, **kwds)