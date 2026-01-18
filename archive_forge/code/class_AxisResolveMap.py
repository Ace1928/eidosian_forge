from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AxisResolveMap(VegaLiteSchema):
    """AxisResolveMap schema wrapper

    Parameters
    ----------

    x : :class:`ResolveMode`, Literal['independent', 'shared']

    y : :class:`ResolveMode`, Literal['independent', 'shared']

    """
    _schema = {'$ref': '#/definitions/AxisResolveMap'}

    def __init__(self, x: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, y: Union['SchemaBase', Literal['independent', 'shared'], UndefinedType]=Undefined, **kwds):
        super(AxisResolveMap, self).__init__(x=x, y=y, **kwds)