from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class RepeatRef(Field):
    """RepeatRef schema wrapper
    Reference to a repeated value.

    Parameters
    ----------

    repeat : Literal['row', 'column', 'repeat', 'layer']

    """
    _schema = {'$ref': '#/definitions/RepeatRef'}

    def __init__(self, repeat: Union[Literal['row', 'column', 'repeat', 'layer'], UndefinedType]=Undefined, **kwds):
        super(RepeatRef, self).__init__(repeat=repeat, **kwds)