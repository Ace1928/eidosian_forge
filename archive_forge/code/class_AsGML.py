from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class AsGML(GeoFunc):
    geom_param_pos = (1,)
    output_field = TextField()

    def __init__(self, expression, version=2, precision=8, **extra):
        expressions = [version, expression]
        if precision is not None:
            expressions.append(self._handle_param(precision, 'precision', int))
        super().__init__(*expressions, **extra)

    def as_oracle(self, compiler, connection, **extra_context):
        source_expressions = self.get_source_expressions()
        version = source_expressions[0]
        clone = self.copy()
        clone.set_source_expressions([source_expressions[1]])
        extra_context['function'] = 'SDO_UTIL.TO_GML311GEOMETRY' if version.value == 3 else 'SDO_UTIL.TO_GMLGEOMETRY'
        return super(AsGML, clone).as_sql(compiler, connection, **extra_context)