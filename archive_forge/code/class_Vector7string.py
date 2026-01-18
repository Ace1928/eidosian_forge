from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class Vector7string(VegaLiteSchema):
    """Vector7string schema wrapper"""
    _schema = {'$ref': '#/definitions/Vector7<string>'}

    def __init__(self, *args):
        super(Vector7string, self).__init__(*args)