from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class RowColnumber(VegaLiteSchema):
    """RowColnumber schema wrapper

    Parameters
    ----------

    column : float

    row : float

    """
    _schema = {'$ref': '#/definitions/RowCol<number>'}

    def __init__(self, column: Union[float, UndefinedType]=Undefined, row: Union[float, UndefinedType]=Undefined, **kwds):
        super(RowColnumber, self).__init__(column=column, row=row, **kwds)