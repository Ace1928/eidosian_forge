from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LegendOrient(VegaLiteSchema):
    """LegendOrient schema wrapper"""
    _schema = {'$ref': '#/definitions/LegendOrient'}

    def __init__(self, *args):
        super(LegendOrient, self).__init__(*args)