from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SelectionConfig(VegaLiteSchema):
    """SelectionConfig schema wrapper

    Parameters
    ----------

    interval : dict, :class:`IntervalSelectionConfigWithoutType`
        The default definition for an `interval
        <https://vega.github.io/vega-lite/docs/parameter.html#select>`__ selection. All
        properties and transformations for an interval selection definition (except ``type``
        ) may be specified here.

        For instance, setting ``interval`` to ``{"translate": false}`` disables the ability
        to move interval selections by default.
    point : dict, :class:`PointSelectionConfigWithoutType`
        The default definition for a `point
        <https://vega.github.io/vega-lite/docs/parameter.html#select>`__ selection. All
        properties and transformations  for a point selection definition (except ``type`` )
        may be specified here.

        For instance, setting ``point`` to ``{"on": "dblclick"}`` populates point selections
        on double-click by default.
    """
    _schema = {'$ref': '#/definitions/SelectionConfig'}

    def __init__(self, interval: Union[dict, 'SchemaBase', UndefinedType]=Undefined, point: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(SelectionConfig, self).__init__(interval=interval, point=point, **kwds)