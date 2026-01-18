from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FieldRange(VegaLiteSchema):
    """FieldRange schema wrapper

    Parameters
    ----------

    field : str

    """
    _schema = {'$ref': '#/definitions/FieldRange'}

    def __init__(self, field: Union[str, UndefinedType]=Undefined, **kwds):
        super(FieldRange, self).__init__(field=field, **kwds)