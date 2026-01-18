from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class DistanceResultMixin:

    @cached_property
    def output_field(self):
        return DistanceField(self.geo_field)

    def source_is_geography(self):
        return self.geo_field.geography and self.geo_field.srid == 4326