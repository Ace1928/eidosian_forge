from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AggregateTransform(Transform):
    """AggregateTransform schema wrapper

    Parameters
    ----------

    aggregate : Sequence[dict, :class:`AggregatedFieldDef`]
        Array of objects that define fields to aggregate.
    groupby : Sequence[str, :class:`FieldName`]
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    """
    _schema = {'$ref': '#/definitions/AggregateTransform'}

    def __init__(self, aggregate: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(AggregateTransform, self).__init__(aggregate=aggregate, groupby=groupby, **kwds)