from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AxisOrient(VegaLiteSchema):
    """AxisOrient schema wrapper"""
    _schema = {'$ref': '#/definitions/AxisOrient'}

    def __init__(self, *args):
        super(AxisOrient, self).__init__(*args)