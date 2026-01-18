from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
@staticmethod
def _from_geojson(geoj):
    shape = Shape()
    geojType = geoj['type'] if geoj else 'Null'
    if geojType == 'Null':
        shapeType = NULL
    elif geojType == 'Point':
        shapeType = POINT
    elif geojType == 'LineString':
        shapeType = POLYLINE
    elif geojType == 'Polygon':
        shapeType = POLYGON
    elif geojType == 'MultiPoint':
        shapeType = MULTIPOINT
    elif geojType == 'MultiLineString':
        shapeType = POLYLINE
    elif geojType == 'MultiPolygon':
        shapeType = POLYGON
    else:
        raise Exception("Cannot create Shape from GeoJSON type '%s'" % geojType)
    shape.shapeType = shapeType
    if geojType == 'Point':
        shape.points = [geoj['coordinates']]
        shape.parts = [0]
    elif geojType in ('MultiPoint', 'LineString'):
        shape.points = geoj['coordinates']
        shape.parts = [0]
    elif geojType in 'Polygon':
        points = []
        parts = []
        index = 0
        for i, ext_or_hole in enumerate(geoj['coordinates']):
            if i == 0 and (not is_cw(ext_or_hole)):
                ext_or_hole = rewind(ext_or_hole)
            elif i > 0 and is_cw(ext_or_hole):
                ext_or_hole = rewind(ext_or_hole)
            points.extend(ext_or_hole)
            parts.append(index)
            index += len(ext_or_hole)
        shape.points = points
        shape.parts = parts
    elif geojType in 'MultiLineString':
        points = []
        parts = []
        index = 0
        for linestring in geoj['coordinates']:
            points.extend(linestring)
            parts.append(index)
            index += len(linestring)
        shape.points = points
        shape.parts = parts
    elif geojType in 'MultiPolygon':
        points = []
        parts = []
        index = 0
        for polygon in geoj['coordinates']:
            for i, ext_or_hole in enumerate(polygon):
                if i == 0 and (not is_cw(ext_or_hole)):
                    ext_or_hole = rewind(ext_or_hole)
                elif i > 0 and is_cw(ext_or_hole):
                    ext_or_hole = rewind(ext_or_hole)
                points.extend(ext_or_hole)
                parts.append(index)
                index += len(ext_or_hole)
        shape.points = points
        shape.parts = parts
    return shape