from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FlattenTransform(Transform):
    """FlattenTransform schema wrapper

    Parameters
    ----------

    flatten : Sequence[str, :class:`FieldName`]
        An array of one or more data fields containing arrays to flatten. If multiple fields
        are specified, their array values should have a parallel structure, ideally with the
        same length. If the lengths of parallel arrays do not match, the longest array will
        be used with ``null`` values added for missing entries.
    as : Sequence[str, :class:`FieldName`]
        The output field names for extracted array values.

        **Default value:** The field name of the corresponding array field
    """
    _schema = {'$ref': '#/definitions/FlattenTransform'}

    def __init__(self, flatten: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(FlattenTransform, self).__init__(flatten=flatten, **kwds)