from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class MultiPoint(Geometry):
    """MultiPoint schema wrapper
    MultiPoint geometry object.  https://tools.ietf.org/html/rfc7946#section-3.1.3

    Parameters
    ----------

    coordinates : Sequence[Sequence[float], :class:`Position`]

    type : str
        Specifies the type of GeoJSON object.
    bbox : :class:`BBox`, Sequence[float]
        Bounding box of the coordinate range of the object's Geometries, Features, or
        Feature Collections. https://tools.ietf.org/html/rfc7946#section-5
    """
    _schema = {'$ref': '#/definitions/MultiPoint'}

    def __init__(self, coordinates: Union[Sequence[Union['SchemaBase', Sequence[float]]], UndefinedType]=Undefined, type: Union[str, UndefinedType]=Undefined, bbox: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, **kwds):
        super(MultiPoint, self).__init__(coordinates=coordinates, type=type, bbox=bbox, **kwds)