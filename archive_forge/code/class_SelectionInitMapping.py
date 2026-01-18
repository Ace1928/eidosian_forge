from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SelectionInitMapping(VegaLiteSchema):
    """SelectionInitMapping schema wrapper"""
    _schema = {'$ref': '#/definitions/SelectionInitMapping'}

    def __init__(self, **kwds):
        super(SelectionInitMapping, self).__init__(**kwds)