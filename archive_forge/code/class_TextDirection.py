from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TextDirection(VegaLiteSchema):
    """TextDirection schema wrapper"""
    _schema = {'$ref': '#/definitions/TextDirection'}

    def __init__(self, *args):
        super(TextDirection, self).__init__(*args)