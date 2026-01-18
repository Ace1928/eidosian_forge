from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class DictSelectionInitInterval(VegaLiteSchema):
    """DictSelectionInitInterval schema wrapper"""
    _schema = {'$ref': '#/definitions/Dict<SelectionInitInterval>'}

    def __init__(self, **kwds):
        super(DictSelectionInitInterval, self).__init__(**kwds)