from django.contrib.gis.gdal import OGRGeomType
from django.db.backends.sqlite3.introspection import (
class GeoFlexibleFieldLookupDict(FlexibleFieldLookupDict):
    """
    Subclass that includes updates the `base_data_types_reverse` dict
    for geometry field types.
    """
    base_data_types_reverse = {**FlexibleFieldLookupDict.base_data_types_reverse, 'point': 'GeometryField', 'linestring': 'GeometryField', 'polygon': 'GeometryField', 'multipoint': 'GeometryField', 'multilinestring': 'GeometryField', 'multipolygon': 'GeometryField', 'geometrycollection': 'GeometryField'}