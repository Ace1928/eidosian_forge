from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LayerRepeatMapping(VegaLiteSchema):
    """LayerRepeatMapping schema wrapper

    Parameters
    ----------

    layer : Sequence[str]
        An array of fields to be repeated as layers.
    column : Sequence[str]
        An array of fields to be repeated horizontally.
    row : Sequence[str]
        An array of fields to be repeated vertically.
    """
    _schema = {'$ref': '#/definitions/LayerRepeatMapping'}

    def __init__(self, layer: Union[Sequence[str], UndefinedType]=Undefined, column: Union[Sequence[str], UndefinedType]=Undefined, row: Union[Sequence[str], UndefinedType]=Undefined, **kwds):
        super(LayerRepeatMapping, self).__init__(layer=layer, column=column, row=row, **kwds)