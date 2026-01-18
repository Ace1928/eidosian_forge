from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class Perimeter(DistanceResultMixin, OracleToleranceMixin, GeoFunc):
    arity = 1

    def as_postgresql(self, compiler, connection, **extra_context):
        function = None
        if self.geo_field.geodetic(connection) and (not self.source_is_geography()):
            raise NotSupportedError('ST_Perimeter cannot use a non-projected non-geography field.')
        dim = min((f.dim for f in self.get_source_fields()))
        if dim > 2:
            function = connection.ops.perimeter3d
        return super().as_sql(compiler, connection, function=function, **extra_context)

    def as_sqlite(self, compiler, connection, **extra_context):
        if self.geo_field.geodetic(connection):
            raise NotSupportedError('Perimeter cannot use a non-projected field.')
        return super().as_sql(compiler, connection, **extra_context)