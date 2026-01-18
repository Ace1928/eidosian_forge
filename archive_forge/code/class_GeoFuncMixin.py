from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class GeoFuncMixin:
    function = None
    geom_param_pos = (0,)

    def __init__(self, *expressions, **extra):
        super().__init__(*expressions, **extra)
        for pos in self.geom_param_pos:
            expr = self.source_expressions[pos]
            if not isinstance(expr, Value):
                continue
            try:
                output_field = expr.output_field
            except FieldError:
                output_field = None
            geom = expr.value
            if not isinstance(geom, GEOSGeometry) or (output_field and (not isinstance(output_field, GeometryField))):
                raise TypeError('%s function requires a geometric argument in position %d.' % (self.name, pos + 1))
            if not geom.srid and (not output_field):
                raise ValueError('SRID is required for all geometries.')
            if not output_field:
                self.source_expressions[pos] = Value(geom, output_field=GeometryField(srid=geom.srid))

    @property
    def name(self):
        return self.__class__.__name__

    @cached_property
    def geo_field(self):
        return self.source_expressions[self.geom_param_pos[0]].field

    def as_sql(self, compiler, connection, function=None, **extra_context):
        if self.function is None and function is None:
            function = connection.ops.spatial_function_name(self.name)
        return super().as_sql(compiler, connection, function=function, **extra_context)

    def resolve_expression(self, *args, **kwargs):
        res = super().resolve_expression(*args, **kwargs)
        if not self.geom_param_pos:
            return res
        source_fields = res.get_source_fields()
        for pos in self.geom_param_pos:
            field = source_fields[pos]
            if not isinstance(field, GeometryField):
                raise TypeError('%s function requires a GeometryField in position %s, got %s.' % (self.name, pos + 1, type(field).__name__))
        base_srid = res.geo_field.srid
        for pos in self.geom_param_pos[1:]:
            expr = res.source_expressions[pos]
            expr_srid = expr.output_field.srid
            if expr_srid != base_srid:
                res.source_expressions[pos] = Transform(expr, base_srid).resolve_expression(*args, **kwargs)
        return res

    def _handle_param(self, value, param_name='', check_types=None):
        if not hasattr(value, 'resolve_expression'):
            if check_types and (not isinstance(value, check_types)):
                raise TypeError('The %s parameter has the wrong type: should be %s.' % (param_name, check_types))
        return value