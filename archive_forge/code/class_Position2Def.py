from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Position2Def(VegaLiteSchema):
    """Position2Def schema wrapper"""
    _schema = {'$ref': '#/definitions/Position2Def'}

    def __init__(self, *args, **kwds):
        super(Position2Def, self).__init__(*args, **kwds)