from django.contrib.gis.db.models.fields import (
from django.db.models import Aggregate, Func, Value
from django.utils.functional import cached_property
class Extent3D(GeoAggregate):
    name = 'Extent3D'
    is_extent = '3D'

    def __init__(self, expression, **extra):
        super().__init__(expression, output_field=ExtentField(), **extra)

    def convert_value(self, value, expression, connection):
        return connection.ops.convert_extent3d(value)