import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _get_geometry_type(gdf):
    """
    Get basic geometry type of a GeoDataFrame. See more info from:
    https://geoalchemy-2.readthedocs.io/en/latest/types.html#geoalchemy2.types._GISType

    Following rules apply:
     - if geometries all share the same geometry-type,
       geometries are inserted with the given GeometryType with following types:
        - Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon,
          GeometryCollection.
        - LinearRing geometries will be converted into LineString -objects.
     - in all other cases, geometries will be inserted with type GEOMETRY:
        - a mix of Polygons and MultiPolygons in GeoSeries
        - a mix of Points and LineStrings in GeoSeries
        - geometry is of type GeometryCollection,
          such as GeometryCollection([Point, LineStrings])
     - if any of the geometries has Z-coordinate, all records will
       be written with 3D.
    """
    geom_types = list(gdf.geometry.geom_type.unique())
    has_curve = False
    for gt in geom_types:
        if gt is None:
            continue
        elif 'LinearRing' in gt:
            has_curve = True
    if len(geom_types) == 1:
        if has_curve:
            target_geom_type = 'LINESTRING'
        elif geom_types[0] is None:
            raise ValueError('No valid geometries in the data.')
        else:
            target_geom_type = geom_types[0].upper()
    else:
        target_geom_type = 'GEOMETRY'
    if any(gdf.geometry.has_z):
        target_geom_type += 'Z'
    return (target_geom_type, has_curve)