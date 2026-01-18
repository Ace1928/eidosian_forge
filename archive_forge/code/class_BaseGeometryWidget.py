import logging
import warnings
from django.conf import settings
from django.contrib.gis import gdal
from django.contrib.gis.geometry import json_regex
from django.contrib.gis.geos import GEOSException, GEOSGeometry
from django.forms.widgets import Widget
from django.utils import translation
from django.utils.deprecation import RemovedInDjango51Warning
class BaseGeometryWidget(Widget):
    """
    The base class for rich geometry widgets.
    Render a map using the WKT of the geometry.
    """
    geom_type = 'GEOMETRY'
    map_srid = 4326
    map_width = 600
    map_height = 400
    display_raw = False
    supports_3d = False
    template_name = ''

    def __init__(self, attrs=None):
        self.attrs = {}
        for key in ('geom_type', 'map_srid', 'map_width', 'map_height', 'display_raw'):
            self.attrs[key] = getattr(self, key)
        if attrs and ('map_width' in attrs or 'map_height' in attrs) or self.map_width != 600 or self.map_height != 400:
            warnings.warn('The map_height and map_width widget attributes are deprecated. Please use CSS to size map widgets.', category=RemovedInDjango51Warning, stacklevel=2)
        if attrs:
            self.attrs.update(attrs)

    def serialize(self, value):
        return value.wkt if value else ''

    def deserialize(self, value):
        try:
            return GEOSGeometry(value)
        except (GEOSException, ValueError, TypeError) as err:
            logger.error("Error creating geometry from value '%s' (%s)", value, err)
        return None

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if value and isinstance(value, str):
            value = self.deserialize(value)
        if value:
            if value.srid and value.srid != self.map_srid:
                try:
                    ogr = value.ogr
                    ogr.transform(self.map_srid)
                    value = ogr
                except gdal.GDALException as err:
                    logger.error("Error transforming geometry from srid '%s' to srid '%s' (%s)", value.srid, self.map_srid, err)
        geom_type = gdal.OGRGeomType(self.attrs['geom_type']).name
        context.update(self.build_attrs(self.attrs, {'name': name, 'module': 'geodjango_%s' % name.replace('-', '_'), 'serialized': self.serialize(value), 'geom_type': 'Geometry' if geom_type == 'Unknown' else geom_type, 'STATIC_URL': settings.STATIC_URL, 'LANGUAGE_BIDI': translation.get_language_bidi(), **(attrs or {})}))
        return context