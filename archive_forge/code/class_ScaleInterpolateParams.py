from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ScaleInterpolateParams(VegaLiteSchema):
    """ScaleInterpolateParams schema wrapper

    Parameters
    ----------

    type : Literal['rgb', 'cubehelix', 'cubehelix-long']

    gamma : float

    """
    _schema = {'$ref': '#/definitions/ScaleInterpolateParams'}

    def __init__(self, type: Union[Literal['rgb', 'cubehelix', 'cubehelix-long'], UndefinedType]=Undefined, gamma: Union[float, UndefinedType]=Undefined, **kwds):
        super(ScaleInterpolateParams, self).__init__(type=type, gamma=gamma, **kwds)