from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FeatureGeometryGeoJsonProperties(VegaLiteSchema):
    """FeatureGeometryGeoJsonProperties schema wrapper
    A feature object which contains a geometry and associated properties.
    https://tools.ietf.org/html/rfc7946#section-3.2

    Parameters
    ----------

    geometry : dict, :class:`Point`, :class:`Polygon`, :class:`Geometry`, :class:`LineString`, :class:`MultiPoint`, :class:`MultiPolygon`, :class:`MultiLineString`, :class:`GeometryCollection`
        The feature's geometry
    properties : dict, None, :class:`GeoJsonProperties`
        Properties associated with this feature.
    type : str
        Specifies the type of GeoJSON object.
    bbox : :class:`BBox`, Sequence[float]
        Bounding box of the coordinate range of the object's Geometries, Features, or
        Feature Collections. https://tools.ietf.org/html/rfc7946#section-5
    id : str, float
        A value that uniquely identifies this feature in a
        https://tools.ietf.org/html/rfc7946#section-3.2.
    """
    _schema = {'$ref': '#/definitions/Feature<Geometry,GeoJsonProperties>'}

    def __init__(self, geometry: Union[dict, 'SchemaBase', UndefinedType]=Undefined, properties: Union[dict, None, 'SchemaBase', UndefinedType]=Undefined, type: Union[str, UndefinedType]=Undefined, bbox: Union['SchemaBase', Sequence[float], UndefinedType]=Undefined, id: Union[str, float, UndefinedType]=Undefined, **kwds):
        super(FeatureGeometryGeoJsonProperties, self).__init__(geometry=geometry, properties=properties, type=type, bbox=bbox, id=id, **kwds)