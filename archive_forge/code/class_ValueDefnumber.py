from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ValueDefnumber(OffsetDef):
    """ValueDefnumber schema wrapper
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Parameters
    ----------

    value : float
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDef<number>'}

    def __init__(self, value: Union[float, UndefinedType]=Undefined, **kwds):
        super(ValueDefnumber, self).__init__(value=value, **kwds)