import re
from django.conf import settings
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.db.models import GeometryField, RasterField
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db import NotSupportedError, ProgrammingError
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.db.models import Func, Value
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
from .adapter import PostGISAdapter
from .models import PostGISGeometryColumns, PostGISSpatialRefSys
from .pgraster import from_pgraster
class PostGISOperator(SpatialOperator):

    def __init__(self, geography=False, raster=False, **kwargs):
        self.geography = geography
        self.raster = raster
        super().__init__(**kwargs)

    def as_sql(self, connection, lookup, template_params, *args):
        template_params = self.check_raster(lookup, template_params)
        template_params = self.check_geography(lookup, template_params)
        return super().as_sql(connection, lookup, template_params, *args)

    def check_raster(self, lookup, template_params):
        spheroid = lookup.rhs_params and lookup.rhs_params[-1] == 'spheroid'
        lhs_is_raster = lookup.lhs.field.geom_type == 'RASTER'
        rhs_is_raster = isinstance(lookup.rhs, GDALRaster)
        if lookup.band_lhs is not None and lhs_is_raster:
            if not self.func:
                raise ValueError('Band indices are not allowed for this operator, it works on bbox only.')
            template_params['lhs'] = '%s, %s' % (template_params['lhs'], lookup.band_lhs)
        if lookup.band_rhs is not None and rhs_is_raster:
            if not self.func:
                raise ValueError('Band indices are not allowed for this operator, it works on bbox only.')
            template_params['rhs'] = '%s, %s' % (template_params['rhs'], lookup.band_rhs)
        if not self.raster or spheroid:
            if lhs_is_raster:
                template_params['lhs'] = 'ST_Polygon(%s)' % template_params['lhs']
            if rhs_is_raster:
                template_params['rhs'] = 'ST_Polygon(%s)' % template_params['rhs']
        elif self.raster == BILATERAL:
            if lhs_is_raster and (not rhs_is_raster):
                template_params['lhs'] = 'ST_Polygon(%s)' % template_params['lhs']
            elif rhs_is_raster and (not lhs_is_raster):
                template_params['rhs'] = 'ST_Polygon(%s)' % template_params['rhs']
        return template_params

    def check_geography(self, lookup, template_params):
        """Convert geography fields to geometry types, if necessary."""
        if lookup.lhs.output_field.geography and (not self.geography):
            template_params['lhs'] += '::geometry'
        return template_params