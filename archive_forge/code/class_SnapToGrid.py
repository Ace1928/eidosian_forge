from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class SnapToGrid(SQLiteDecimalToFloatMixin, GeomOutputGeoFunc):

    def __init__(self, expression, *args, **extra):
        nargs = len(args)
        expressions = [expression]
        if nargs in (1, 2):
            expressions.extend([self._handle_param(arg, '', NUMERIC_TYPES) for arg in args])
        elif nargs == 4:
            expressions += [*(self._handle_param(arg, '', NUMERIC_TYPES) for arg in args[2:]), *(self._handle_param(arg, '', NUMERIC_TYPES) for arg in args[0:2])]
        else:
            raise ValueError('Must provide 1, 2, or 4 arguments to `SnapToGrid`.')
        super().__init__(*expressions, **extra)