from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class MarkType(VegaLiteSchema):
    """MarkType schema wrapper"""
    _schema = {'$ref': '#/definitions/MarkType'}

    def __init__(self, *args):
        super(MarkType, self).__init__(*args)