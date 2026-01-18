from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ColorName(Color):
    """ColorName schema wrapper"""
    _schema = {'$ref': '#/definitions/ColorName'}

    def __init__(self, *args):
        super(ColorName, self).__init__(*args)