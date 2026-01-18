from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class UtcMultiTimeUnit(MultiTimeUnit):
    """UtcMultiTimeUnit schema wrapper"""
    _schema = {'$ref': '#/definitions/UtcMultiTimeUnit'}

    def __init__(self, *args):
        super(UtcMultiTimeUnit, self).__init__(*args)