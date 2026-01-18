from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class StyleConfigIndex(VegaLiteSchema):
    """StyleConfigIndex schema wrapper

    Parameters
    ----------

    arc : dict, :class:`RectConfig`
        Arc-specific Config
    area : dict, :class:`AreaConfig`
        Area-Specific Config
    bar : dict, :class:`BarConfig`
        Bar-Specific Config
    circle : dict, :class:`MarkConfig`
        Circle-Specific Config
    geoshape : dict, :class:`MarkConfig`
        Geoshape-Specific Config
    image : dict, :class:`RectConfig`
        Image-specific Config
    line : dict, :class:`LineConfig`
        Line-Specific Config
    mark : dict, :class:`MarkConfig`
        Mark Config
    point : dict, :class:`MarkConfig`
        Point-Specific Config
    rect : dict, :class:`RectConfig`
        Rect-Specific Config
    rule : dict, :class:`MarkConfig`
        Rule-Specific Config
    square : dict, :class:`MarkConfig`
        Square-Specific Config
    text : dict, :class:`MarkConfig`
        Text-Specific Config
    tick : dict, :class:`TickConfig`
        Tick-Specific Config
    trail : dict, :class:`LineConfig`
        Trail-Specific Config
    group-subtitle : dict, :class:`MarkConfig`
        Default style for chart subtitles
    group-title : dict, :class:`MarkConfig`
        Default style for chart titles
    guide-label : dict, :class:`MarkConfig`
        Default style for axis, legend, and header labels.
    guide-title : dict, :class:`MarkConfig`
        Default style for axis, legend, and header titles.
    """
    _schema = {'$ref': '#/definitions/StyleConfigIndex'}

    def __init__(self, arc: Union[dict, 'SchemaBase', UndefinedType]=Undefined, area: Union[dict, 'SchemaBase', UndefinedType]=Undefined, bar: Union[dict, 'SchemaBase', UndefinedType]=Undefined, circle: Union[dict, 'SchemaBase', UndefinedType]=Undefined, geoshape: Union[dict, 'SchemaBase', UndefinedType]=Undefined, image: Union[dict, 'SchemaBase', UndefinedType]=Undefined, line: Union[dict, 'SchemaBase', UndefinedType]=Undefined, mark: Union[dict, 'SchemaBase', UndefinedType]=Undefined, point: Union[dict, 'SchemaBase', UndefinedType]=Undefined, rect: Union[dict, 'SchemaBase', UndefinedType]=Undefined, rule: Union[dict, 'SchemaBase', UndefinedType]=Undefined, square: Union[dict, 'SchemaBase', UndefinedType]=Undefined, text: Union[dict, 'SchemaBase', UndefinedType]=Undefined, tick: Union[dict, 'SchemaBase', UndefinedType]=Undefined, trail: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(StyleConfigIndex, self).__init__(arc=arc, area=area, bar=bar, circle=circle, geoshape=geoshape, image=image, line=line, mark=mark, point=point, rect=rect, rule=rule, square=square, text=text, tick=tick, trail=trail, **kwds)