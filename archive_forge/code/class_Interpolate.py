from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Interpolate(VegaLiteSchema):
    """Interpolate schema wrapper"""
    _schema = {'$ref': '#/definitions/Interpolate'}

    def __init__(self, *args):
        super(Interpolate, self).__init__(*args)