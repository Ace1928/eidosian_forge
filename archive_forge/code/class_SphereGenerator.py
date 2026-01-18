from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SphereGenerator(Generator):
    """SphereGenerator schema wrapper

    Parameters
    ----------

    sphere : bool, dict
        Generate sphere GeoJSON data for the full globe.
    name : str
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/SphereGenerator'}

    def __init__(self, sphere: Union[bool, dict, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(SphereGenerator, self).__init__(sphere=sphere, name=name, **kwds)