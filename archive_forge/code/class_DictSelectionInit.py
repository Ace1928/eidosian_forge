from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DictSelectionInit(VegaLiteSchema):
    """DictSelectionInit schema wrapper"""
    _schema = {'$ref': '#/definitions/Dict<SelectionInit>'}

    def __init__(self, **kwds):
        super(DictSelectionInit, self).__init__(**kwds)