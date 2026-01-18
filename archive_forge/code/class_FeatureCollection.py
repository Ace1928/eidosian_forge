from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FeatureCollection(VegaLiteSchema):
    """FeatureCollection schema wrapper
    A collection of feature objects.  https://tools.ietf.org/html/rfc7946#section-3.3

    Parameters
    ----------

    features : Sequence[dict, :class:`FeatureGeometryGeoJsonProperties`]

    type : str
        Specifies the type of GeoJSON object.
    bbox : :class:`BBox`, Sequence[float]
        Bounding box of the coordinate range of the object's Geometries, Features, or
        Feature Collections. https://tools.ietf.org/html/rfc7946#section-5
    """
    _schema = {'$ref': '#/definitions/FeatureCollection'}

    def __init__(self, features: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, type: Union[str, UndefinedType]=Undefined, bbox: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, **kwds):
        super(FeatureCollection, self).__init__(features=features, type=type, bbox=bbox, **kwds)