from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class AsSVG(GeoFunc):
    output_field = TextField()

    def __init__(self, expression, relative=False, precision=8, **extra):
        relative = relative if hasattr(relative, 'resolve_expression') else int(relative)
        expressions = [expression, relative, self._handle_param(precision, 'precision', int)]
        super().__init__(*expressions, **extra)