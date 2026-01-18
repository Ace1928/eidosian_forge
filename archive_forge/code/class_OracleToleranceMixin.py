from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class OracleToleranceMixin:
    tolerance = 0.05

    def as_oracle(self, compiler, connection, **extra_context):
        tolerance = Value(self._handle_param(self.extra.get('tolerance', self.tolerance), 'tolerance', NUMERIC_TYPES))
        clone = self.copy()
        clone.set_source_expressions([*self.get_source_expressions(), tolerance])
        return clone.as_sql(compiler, connection, **extra_context)