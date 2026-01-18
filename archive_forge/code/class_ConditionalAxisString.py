from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ConditionalAxisString(VegaLiteSchema):
    """ConditionalAxisString schema wrapper"""
    _schema = {'$ref': '#/definitions/ConditionalAxisString'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisString, self).__init__(*args, **kwds)