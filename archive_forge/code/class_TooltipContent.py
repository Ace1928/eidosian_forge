from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TooltipContent(VegaLiteSchema):
    """TooltipContent schema wrapper

    Parameters
    ----------

    content : Literal['encoding', 'data']

    """
    _schema = {'$ref': '#/definitions/TooltipContent'}

    def __init__(self, content: Union[Literal['encoding', 'data'], UndefinedType]=Undefined, **kwds):
        super(TooltipContent, self).__init__(content=content, **kwds)