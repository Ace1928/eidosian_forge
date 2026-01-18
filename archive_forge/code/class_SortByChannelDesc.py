from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SortByChannelDesc(AllSortString):
    """SortByChannelDesc schema wrapper"""
    _schema = {'$ref': '#/definitions/SortByChannelDesc'}

    def __init__(self, *args):
        super(SortByChannelDesc, self).__init__(*args)