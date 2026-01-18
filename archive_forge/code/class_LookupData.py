from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LookupData(VegaLiteSchema):
    """LookupData schema wrapper

    Parameters
    ----------

    data : dict, :class:`Data`, :class:`UrlData`, :class:`Generator`, :class:`NamedData`, :class:`DataSource`, :class:`InlineData`, :class:`SphereGenerator`, :class:`SequenceGenerator`, :class:`GraticuleGenerator`
        Secondary data source to lookup in.
    key : str, :class:`FieldName`
        Key in data to lookup.
    fields : Sequence[str, :class:`FieldName`]
        Fields in foreign data or selection to lookup. If not specified, the entire object
        is queried.
    """
    _schema = {'$ref': '#/definitions/LookupData'}

    def __init__(self, data: Union[dict, 'SchemaBase', UndefinedType]=Undefined, key: Union[str, 'SchemaBase', UndefinedType]=Undefined, fields: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(LookupData, self).__init__(data=data, key=key, fields=fields, **kwds)