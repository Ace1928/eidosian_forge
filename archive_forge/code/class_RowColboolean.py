from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class RowColboolean(VegaLiteSchema):
    """RowColboolean schema wrapper

    Parameters
    ----------

    column : bool

    row : bool

    """
    _schema = {'$ref': '#/definitions/RowCol<boolean>'}

    def __init__(self, column: Union[bool, UndefinedType]=Undefined, row: Union[bool, UndefinedType]=Undefined, **kwds):
        super(RowColboolean, self).__init__(column=column, row=row, **kwds)