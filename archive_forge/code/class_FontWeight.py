from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FontWeight(VegaLiteSchema):
    """FontWeight schema wrapper"""
    _schema = {'$ref': '#/definitions/FontWeight'}

    def __init__(self, *args):
        super(FontWeight, self).__init__(*args)