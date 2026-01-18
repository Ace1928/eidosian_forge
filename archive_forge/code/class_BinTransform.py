from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BinTransform(Transform):
    """BinTransform schema wrapper

    Parameters
    ----------

    bin : bool, dict, :class:`BinParams`
        An object indicating bin properties, or simply ``true`` for using default bin
        parameters.
    field : str, :class:`FieldName`
        The data field to bin.
    as : str, :class:`FieldName`, Sequence[str, :class:`FieldName`]
        The output fields at which to write the start and end bin values. This can be either
        a string or an array of strings with two elements denoting the name for the fields
        for bin start and bin end respectively. If a single string (e.g., ``"val"`` ) is
        provided, the end field will be ``"val_end"``.
    """
    _schema = {'$ref': '#/definitions/BinTransform'}

    def __init__(self, bin: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, field: Union[str, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(BinTransform, self).__init__(bin=bin, field=field, **kwds)